# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:40:18 2021

@author: QYK
"""


import os
import pandas as pd 
import nibabel as nib
from mri.analysis.prepare_data import smoothData
from nilearn.image import load_img


if __name__ =="__main__":
    data_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/fmriprep'
    subjects = ['024']
    subjects.sort()
    func_name = r'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    
    run = [1,2,3,4,5,6]
    for subj in subjects:
        # --------可替换的脚本----------#
        smoothData(subj,run,func_name)