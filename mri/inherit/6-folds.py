# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:08:43 2021

@author: QYK
"""
import os
import numpy as np
import nibabel as nib
import pandas as pd 
from scipy.stats import circmean
from nilearn.image import math_img,concat_imgs,mean_img,resample_to_img
from nilearn.masking import apply_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix

import matplotlib.pyplot as plt
from simulation import plot
#%%
# estimate the omega
# concat the events 
subj = 'sub-005'
tr = 3
tr_number = 134
run_list = range(1,10)
ifold_list = range(1,10)
paramEsti = {'ifold':[],'mean_orientation':[], 'beta_estimate':[]}
columns = ['ifold','mean_orientation', 'beta_estimate']


event_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-005\events\hexagon_on_M2'
event_train = []
for i,run_id in enumerate(run_list):
    event_path = os.path.join(event_dir,'{}_task-game1_run-{}_events.tsv'.format(subj,run_id))
    event = pd.read_csv(event_path,sep='\t')
    event['onset'] = event['onset'] + i * 134 * 3
    event_train.append(event)
event_train = pd.concat(event_train)

# concat the fMRI data
func_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\func'.format(subj)
func_1to7 = []
for i,run_id in enumerate(run_list):
    dm_path = os.path.join(func_dir,'{}_task-game1_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(subj,run_id))
    func_1to7.append(dm_path)
func_1to7 = concat_imgs(func_1to7)


# generate design matrixs
motion = []
add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
for run_id in run_list:
    confound_file = os.path.join(func_dir,'{}_task-game1_run-{}_desc-confounds_timeseries.tsv'.format(subj,run_id))
    confound_factors = pd.read_csv(confound_file,sep="\t")
    motion.append(confound_factors[add_reg_names])
motion = pd.concat(motion,axis=0)

# generate the hrf design matrix
n_scans = tr_number * len(run_list)
frame_times = np.arange(n_scans) * tr
# generate the design matrix with head motion
design_matrix = make_first_level_design_matrix(frame_times, 
    event_train, drift_model=None, add_regs=motion,
    add_reg_names=add_reg_names,hrf_model='spm')


# add run label
run = []
for i in run_list:
    run_id = [i]*tr_number
    run.extend(run_id)
design_matrix['run'] =run


# estimate_omega map

fmri_glm = FirstLevelModel(smoothing_fwhm=12,slice_time_ref=0.5)
fmri_glm = fmri_glm.fit(func_1to7, design_matrices=design_matrix)

# define contrast
contrast_matrix = np.eye(design_matrix.shape[1])
basic_contrasts = dict([(column, contrast_matrix[i])
                for i, column in enumerate(design_matrix.columns)])

# beta_cos and beta_sin map
beta_sin_map = fmri_glm.compute_contrast(basic_contrasts['sin'],output_type='effect_size')
beta_cos_map = fmri_glm.compute_contrast(basic_contrasts['cos'],output_type='effect_size')
fai_rad_map = math_img("np.arctan(img1/img2)",img1=beta_sin_map,img2=beta_cos_map)

# extract omeag from mask and calculate the mean orientation
#ec_mask = nib.load(r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\mask/{}_EC_roi.nii.gz'.format(subj,subj))
ec_mask = nib.load(r'D:\Data\Template/EC_anatomy_roi.nii.gz')
ec_mask = resample_to_img(ec_mask,fai_rad_map,interpolation='nearest')
fai_rad_map_ec = apply_mask(fai_rad_map,ec_mask)
#%%
ifold_orientation = dict()
for ifold in ifold_list:    
    mean_orientation = np.rad2deg(circmean(fai_rad_map_ec)/ifold)
    # mean_oritation = np.rad2deg(fai_rad_map_ec.mean())/ifold
    ifold_orientation[ifold] = mean_orientation

    
    # concat test run
    test_run = run_list
    event_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\events\hexagon_on_M2'.format(subj)
    event_test = []
    for i,run_id in enumerate(test_run):
        event_path = os.path.join(event_dir,'{}_task-game1_run-{}_events.tsv'.format(subj,run_id))
        event = pd.read_csv(event_path,sep='\t')
        event['onset'] = event['onset'] + i*134*3
        event_test.append(event)
    event_test = pd.concat(event_test)
    
    # concat the fMRI data
    func_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\func'.format(subj)
    func_test = []
    for i,run_id in enumerate(test_run):
        dm_path = os.path.join(func_dir,'{}_task-game1_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(subj,run_id))
        func_test.append(dm_path)
    func_test = concat_imgs(func_test)
    
    # generate test design matrixs
    cos_subtract_fai = event_test.loc[event_test.trial_type=='cos']
    event_test = event_test.drop(event_test.loc[event_test.trial_type=='sin'].index)
    event_test = event_test.drop(event_test.loc[event_test.trial_type=='cos'].index)
    
    angle = cos_subtract_fai['angle']
    cos_subtract_fai['trial_type'] = 'cos_subtract_fai'
    cos_subtract_fai['modulation'] = np.cos(np.deg2rad(ifold*(angle - mean_orientation)))
    event_test = pd.concat([event_test,cos_subtract_fai],axis=0)
    
    # generate the hrf design matrix
    n_scans = tr_number  * len(test_run)
    frame_times = np.arange(n_scans) * tr
    # generate the design matrix with head motion
    design_matrix = make_first_level_design_matrix(frame_times, 
        event_test, drift_model=None, add_regs=motion, 
        add_reg_names=add_reg_names,hrf_model='spm')
    # add run label
    run = []
    for i in test_run:
        run_id = [i]* tr_number 
        run.extend(run_id)
    design_matrix['run'] = run

    # estimate the beta of cos subtract fai
    fmri_glm = FirstLevelModel(smoothing_fwhm=12,slice_time_ref=0.5)
    fmri_glm = fmri_glm.fit(func_1to7, design_matrices=design_matrix)
    
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                    for i, column in enumerate(design_matrix.columns)])
    beta_map = fmri_glm.compute_contrast(basic_contrasts['cos_subtract_fai'],output_type='effect_size')
    #ec_mask = nib.load(r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\mask/{}_EC_roi.nii.gz'.format(subj,subj))
    ec_mask = nib.load(r'D:\Data\Template/EC_anatomy_roi.nii.gz')
    ec_mask = resample_to_img(ec_mask,beta_map,interpolation='nearest')
    beta_map_ec = apply_mask(beta_map,ec_mask)
    beta_estimate = beta_map_ec.mean()
    for column in columns:
        paramEsti[column].append(eval(column))
paramEsti_df = pd.DataFrame(paramEsti)
paramEsti_df.set_index('ifold',inplace=True)
paramEsti_df.to_csv(r'D:\Data\Development_Cognitive_Map\bids\derivatives\analysis\glm1\result\sub-005/old_ver_omega_estimate.csv')
plot.plotAngleRadar(angle)
plot.plot6foldSpecificity(paramEsti_df)
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