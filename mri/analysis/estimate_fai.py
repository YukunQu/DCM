# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:25:41 2021

@author: QYK
"""

import os
import numpy as np
from mri.analysis.prepare_data import prepare_data
from mri.analysis.glm import glm1_estimate_fai
from mri.analysis.utils import estimateMeanOrientation_Tim
from nilearn.image import load_img


def estimate_fai(subj,runs,mask_path,save_dir):
    
    # estimate mean orientations of different folds 
    # subj:string, subject id such as 'sub-003'
    # runs:list, a list to indicate which runs will be uesd to estimate fai
    # mask:string, file path to indicate a mask.nii.gz file
    # save_dir:string 
    
    # set functional data name,events file name and some basic parameter
    func_name = 'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    events_name = 'hexonM2Long/{}fold/sub-{}_task-game1_run-{}_events.tsv'
    motion_name = 'sub-{}_task-game1_run-{}_desc-confounds_timeseries.tsv'
    tr = 3 
    
    ifold_fai = {}
    for ifold in range(4,9):
        func_all, design_matrices = prepare_data(subj, runs, func_name, 
                                                 events_name, motion_name,tr,ifold,True)
        
        fmap, beta_sin_map, beta_cos_map = glm1_estimate_fai(func_all,design_matrices)
        
        save_path = os.path.join(save_dir,'{}fold'.format(ifold))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        save_path = os.path.join(save_dir,'{}fold'.format(ifold),'ftest_map.nii.gz')
        fmap.to_filename(save_path)
        
        save_path = os.path.join(save_dir,'{}fold'.format(ifold),'beta_sin_map.nii.gz')
        beta_sin_map.to_filename(save_path)
        
        save_path = os.path.join(save_dir,'{}fold'.format(ifold),'beta_cos_map.nii.gz')
        beta_cos_map.to_filename(save_path)
        
        mask_img = load_img(mask_path)
        mean_orientation =  estimateMeanOrientation_Tim(beta_sin_map, beta_cos_map, mask_img, ifold)
        ifold_fai[ifold] = mean_orientation
            
    np.save(os.path.join(save_dir,'ifold_fat.npy'),ifold_fai)
    return ifold_fai