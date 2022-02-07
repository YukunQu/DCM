# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:25:41 2021

@author: QYK
"""

import os
import numpy as np
from mri.analysis.prepare_data import prepare_data
from mri.analysis.glm import glm2_estimate_hexagon_effect
from nilearn.image import load_img,resample_to_img
from nilearn.masking import apply_mask
import seaborn as sns
import matplotlib.pyplot as plt


def plot6foldSpecificity(paramEsti_df,save_dir):
    sns.set_context(context="talk")
    x = []
    y = []
    for key,value in paramEsti_df.items():
        x.append(key)
        y.append(value)
    sns.barplot(x=x, y=y, palette="rocket")
    plt.xlabel('i-fold')
    plt.ylabel('Beta estimate')
    savepath = os.path.join(save_dir, '6fold-specificity.png')
    plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)
    plt.show()
    

def estimate_hexagon_effect(subj,runs,mask_path,save_dir):
    
    func_name = 'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    events_name = 'fai_effect/{}fold/sub-{}_task-game1_run-{}_events_fai_effect.tsv'
    motion_name = 'sub-{}_task-game1_run-{}_desc-confounds_timeseries.tsv'
    tr = 3 
    
    ifold_beta = {}
    for ifold in range(4,9):
        func_all, design_matrices = prepare_data(subj, runs, func_name, 
                                                 events_name, motion_name, tr, ifold,True)
        
        z_map = glm2_estimate_hexagon_effect(func_all,design_matrices)
        
        save_path = os.path.join(save_dir,'{}fold'.format(ifold))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        save_path = os.path.join(save_dir,'{}fold'.format(ifold),'beta_fai_zmap.nii.gz')
        z_map.to_filename(save_path)
        
        mask_img = load_img(mask_path)
        mask_img = resample_to_img(mask_img,z_map,interpolation='nearest')
        z_map_masked = apply_mask(z_map,mask_img)
        beta_estimate = z_map_masked.mean()
        ifold_beta[ifold] = beta_estimate
    
    np.save(os.path.join(save_dir,'ifold_beta_fai.npy'),ifold_beta)
    plot6foldSpecificity(ifold_beta,save_dir)