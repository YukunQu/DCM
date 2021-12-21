# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:08:43 2021

@author: QYK
"""
import os
from os.path import join
import numpy as np
import nibabel as nib
import pandas as pd 
from scipy.stats import circmean
from nilearn.image import math_img,concat_imgs,mean_img,index_img,resample_to_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.masking import apply_mask

from simulation import plot
import matplotlib.pyplot as plt

import seaborn as sns
#%%
# estimate the omega
# concat the events 
subj = 'sub-005'
run_list = range(1,10)
ifold_list = range(1,10)
# paramEsti = {'ifold':[],'mean_oritation':[], 'beta_estimate':[]}
# columns = ['ifold','mean_oritation', 'beta_estimate']
ifold_label = []
mean_oritation_list = []
beta_estimate = []
#%%
# estimate the beta_sin map and beta_cos map

# load train event file 
event_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\events\hexagon_on_M2'.format(subj)
train_event_path = join(event_dir,'sub-005_task-game1_train_fai_events.tsv')
train_event = pd.read_csv(train_event_path,sep='\t')

# concat the fMRI data
func_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\func'.format(subj)
func_concat = []
for i,run_id in enumerate(run_list):
    dm_path = os.path.join(func_dir,'{}_task-game1_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(subj,run_id))
    func_concat.append(dm_path)
func_concat = concat_imgs(func_concat)

# generate design matrixs
motion = []
for run_id in run_list:
    add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound_file = os.path.join(func_dir,'{}_task-game1_run-{}_desc-confounds_timeseries.tsv'.format(subj,run_id))
    confound_factors = pd.read_csv(confound_file,sep="\t")
    motion_part = confound_factors[add_reg_names]
    motion.append(motion_part)
motion = pd.concat(motion,axis=0)

# generate the hrf design matrix
tr = 3
n_scans = 134 * len(run_list)
frame_times = np.arange(n_scans) * tr
# generate the design matrix with head motion
design_matrix = make_first_level_design_matrix(frame_times, 
                train_event, drift_model=None, add_regs=motion, 
                add_reg_names=add_reg_names,hrf_model='spm')
# add run label
run = []
for i in run_list:
    run_id = [i]*134
    run.extend(run_id)
design_matrix['run'] =run

# estimate_omega map
fmri_glm = FirstLevelModel(smoothing_fwhm=12)
fmri_glm = fmri_glm.fit(func_concat, design_matrices=design_matrix)

# define contrast
contrast_matrix = np.eye(design_matrix.shape[1])
basic_contrasts = dict([(column, contrast_matrix[i])
                for i, column in enumerate(design_matrix.columns)])

# beta_cos and beta_sin map
beta_sin_map = fmri_glm.compute_contrast(basic_contrasts['sin'],output_type='effect_size')
beta_cos_map = fmri_glm.compute_contrast(basic_contrasts['cos'],output_type='effect_size')
#%%
run_oirentation = {}
for ifold in ifold_list:    
    # extract omeag from mask and calculate the mean orientation
    fai_rad_map = math_img("np.arctan(img1/img2)",img1=beta_sin_map,img2=beta_cos_map)
    ec_mask = nib.load(r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-005\mask/sub-005_func_EC_roi.nii.gz')
    ec_mask = resample_to_img(ec_mask,fai_rad_map,interpolation='nearest')
    fai_rad_map_ec = apply_mask(fai_rad_map,ec_mask)
    mean_oritation = np.rad2deg(circmean(fai_rad_map_ec)/ifold)

    # concat test run
    test_event_path = join(event_dir,'sub-005_task-game1_test_fai_events.tsv')
    test_event = pd.read_csv(test_event_path,sep='\t')
    #%%
    # generate test design matrixs    
    angle = test_event['angle']
    test_event['modulation'] = np.cos(np.deg2rad(ifold*(angle - mean_oritation)))
    
    # generate the hrf design matrix
    tr = 3
    n_scans = 134 * len(run_list)
    frame_times = np.arange(n_scans) * tr
    # generate the design matrix with head motion
    design_matrix = make_first_level_design_matrix(frame_times, 
        test_event, drift_model=None, add_regs=motion, 
        add_reg_names=add_reg_names,hrf_model='spm')

    # add run label
    run = []
    for i in run_list:
        run_id = [i]*134
        run.extend(run_id)
    design_matrix['run'] = run
    
    # estimate the beta of esimateed mean orientaion
    fmri_glm = FirstLevelModel(smoothing_fwhm=12)
    fmri_glm = fmri_glm.fit(func_concat, design_matrices=design_matrix)
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                    for i, column in enumerate(design_matrix.columns)])
    beta_map = fmri_glm.compute_contrast(basic_contrasts['cos_subtract_fai'],output_type='effect_size')
    beta_ec = apply_mask(beta_map, ec_mask)
    
    volex_number = len(beta_ec) 
    ifold_label.extend(volex_number*[ifold])
    mean_oritation_list.extend(volex_number*[mean_oritation])
    beta_estimate.extend(list(beta_ec))
#%%
result = pd.DataFrame({'ifold':ifold_label,'beta_estimate':beta_estimate})
sns.catplot(data=result,x='ifold',y='beta_estimate')
plt.show()
#%%
    # # extract mean activity on stimuli condition
    # m2_event = event_7to9[event_7to9['trial_type']=='M2']
    # angle = m2_event['angle']
    # volume_index = np.ceil(m2_event['onset']/3).astype('int').to_list()
    # m2_activation = index_img(func_7to9,volume_index)
    # masker = NiftiMasker(standardize=True, mask_img=ec_mask)
    # m2_mean_activation_ec = masker.fit_transform(m2_activation).mean(axis=1)
    # testAngles = angle
    # testActivation = m2_mean_activation_ec
    # plot.alignVSmisalign(testAngles, testActivation, mean_oritation)

