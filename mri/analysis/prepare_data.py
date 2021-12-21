# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:24:59 2021

@author: QYK
"""
import os
import numpy as np
import pandas as pd 
from os.path import join
from nilearn.image import smooth_img
from nilearn.image import load_img,concat_imgs
from nilearn.glm.first_level import make_first_level_design_matrix




def prepare_data(subj,run_list,func_name,events_name,motion_name,tr,ifold,fixed_effect=True):
    """concatenate images and design matrixs from different run """
    
    func_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/fmriprep/sub-{}/func'.format(subj)
    event_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/fmriprep/sub-{}/events'.format(subj)        
    func_all = []
    design_matrices = []
    for i,run_id in enumerate(run_list):
        # load image
        func_path = join(func_dir,func_name.format(subj,run_id))
        func_img = load_img(func_path)
        func_all.append(func_img)
        
        # load event
        n_scans = func_img.shape[-1]
        frame_times = np.arange(n_scans) * tr
        event_path = join(event_dir,events_name.format(ifold,subj,run_id))
        print(event_path)
        event = pd.read_csv(event_path,sep='\t')
        
        # load motion
        add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        confound_file = os.path.join(func_dir,motion_name.format(subj,run_id))
        confound_factors = pd.read_csv(confound_file,sep="\t")
        motion = confound_factors[add_reg_names]

        # creat design matrix
        high_pass_fre = 1/128
        design_matrix = make_first_level_design_matrix(
                frame_times,
                event,
                hrf_model='spm',
                drift_model=None,
                high_pass=high_pass_fre,add_regs=motion,add_reg_names=add_reg_names
                )
        design_matrices.append(design_matrix)
        
    if not fixed_effect:
        func_all = concat_imgs(func_all)
        design_matrices = pd.concat(design_matrices,axis=0)
    return func_all, design_matrices
    

def smoothData(subj,run_list,func_name):
    func_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-{}\func'.format(subj)

    for i,run_id in enumerate(run_list):
        # load image
        func_path = join(func_dir,func_name.format(subj,run_id))
        func_img = load_img(func_path)
        func_img = smooth_img(func_path,8)
        save_path = os.path.join(func_dir,'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth8.nii.gz'.format(subj,run_id))
        func_img.to_filename(save_path)
        print('Output:sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth8.nii.gz'.format(subj,run_id))
