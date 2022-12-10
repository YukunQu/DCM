# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:09:00 2022

@author: QYK
"""
import pandas as pd
from scipy.io import loadmat
from nilearn.image import mean_img,math_img


class SPMfile:
    
    def __init__(self,filepath):
        self.filepath = filepath
        self.spmmat = self.load_spm()
        self.design_matrix = self.load_spm_dm()
        
    def load_spm(self):
        return loadmat(self.filepath, struct_as_record=False)
    
    def load_spm_dm(self):
        designMatrix = self.spmmat['SPM'][0][0].xX[0][0].X
        names = [i[0] for i in self.spmmat['SPM'][0][0].xX[0][0].name[0]]
        design_matrix = pd.DataFrame(designMatrix,columns=names)
        return design_matrix
    
    def get_reg_index(self,target_name):
        target_index = []
        for i,reg_name in enumerate(self.design_matrix.columns):
            if target_name in reg_name:
                target_index.append(i+1)
        return target_index
                    
    def get_regs_index(self,target_names):
        # reg_names: a list of regressor names
        targets_index = []
        for target_name in target_names:
            target_index = self.get_reg_index(target_name)
            targets_index.append(target_index)
        return targets_index


def get_averge_bmap(bmap_template,indexs):
    beta_maps = []
    for i in indexs:
        beta_maps.append(bmap_template.format(str(i).zfill(4)))
    beta_average_map = mean_img(beta_maps)
    return beta_average_map


# peri_stimulus
# load spm files
spm_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/work_1st/_subj_id_187/level1conest/SPM.mat'
spmfile = SPMfile(spm_path)
# get index from spm file
cos_regs_names = ['infer_corrxcos^1*bf({})'.format(i) for i in range(1,6)]  # bf(1) ~ bf(5)
sin_regs_names = ['infer_corrxsin^1*bf({})'.format(i) for i in range(1,6)]
cos_regs_index = spmfile.get_regs_index(cos_regs_names)
sin_regs_index = spmfile.get_regs_index(sin_regs_names)
#%%
# extract beta images
beta_template = r'D:\Project\Development_cognitive_map\Data\BIDS\derivatives\Nipype\fir_hexagon\sub-180\level1estimate\beta_{}.nii'
cos_peri_stimuli_maps = [get_averge_bmap(beta_template,index) for index in cos_regs_index]
sin_peri_stimuli_maps = [get_averge_bmap(beta_template,index) for index in sin_regs_index]

# get hexagonal peri stimuli maps
for i,(cos_map,sin_map) in enumerate(zip(cos_peri_stimuli_maps,sin_peri_stimuli_maps)):
    hexagonal_peri_stimuli_map = (math_img("np.sqrt(img1**2+img2**2)",img1=cos_map,img2=sin_map))
    print(hexagonal_peri_stimuli_map.get_fdata().mean(),hexagonal_peri_stimuli_map.get_fdata().std())
    savepath =r'D:\Project\Development_cognitive_map\Data\BIDS\derivatives\Nipype\fir_hexagon\sub-180\level1estimate\hexagonal_peri_stimuli_map{}.nii.gz'.format(i+1)
    hexagonal_peri_stimuli_map.to_filename(savepath)