#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 00:03:50 2022

@author: dell
"""
import os
import numpy as np
from nilearn.image import load_img,math_img,new_img_like
from  nilearn.masking import apply_mask


def zscore_nii(source_dir,file,prefix):
    filepath = os.path.join(source_dir,file)
    mask = load_img(r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')
    #mask = load_img(r'/mnt/data/Template/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz')
    fmap = load_img(filepath)
    
    fmap_mean = apply_mask(fmap, mask).mean()
    fmap_std = apply_mask(fmap, mask).std()
    fmap_zscore = math_img("(img- {})/{}".format(fmap_mean,fmap_std),img=fmap)
    
    fmap_data = fmap_zscore.get_fdata()
    mask_data = mask.get_fdata()
    fmap_data[mask_data == 0] = np.float64('NaN')
    map_zscore = new_img_like(fmap_zscore, fmap_data)
    map_zscore.to_filename(os.path.join(source_dir,prefix+file[3:]))

#%%
# zscore the 1st F-test result
testset_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/separate_hexagon/Setall'

for ifold in range(6,7):
    ifold = f'{ifold}fold'
    data_dir = os.path.join(testset_dir,ifold)
    sub_list = os.listdir(data_dir)
    for sub in sub_list:
        data_sub_dir = os.path.join(data_dir,sub)
        #zscore_nii(data_sub_dir, 'spmF_0004.nii', 'Z')
        zscore_nii(data_sub_dir, 'spmF_0005.nii', 'Z')
        zscore_nii(data_sub_dir, 'spmF_0006.nii', 'Z')
        zscore_nii(data_sub_dir, 'spmT_0007.nii', 'Z')
        zscore_nii(data_sub_dir, 'spmT_0008.nii', 'Z')
        zscore_nii(data_sub_dir, 'spmF_0011.nii', 'Z')
        print(sub,'was zscored.')

#%%
# zscore cross validation result
testset_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/alignPhi/EC_group/Set{}'

for set_id in [1,2]:
    for ifold in range(6,7):
        ifold = f'{ifold}fold'
        data_dir = os.path.join(testset_dir.format(set_id),ifold)
        sub_list = os.listdir(data_dir)
        for sub in sub_list:
            data_sub_dir = os.path.join(data_dir,sub)
            zscore_nii(data_sub_dir, 'spmT_0001.nii', 'Z')
            zscore_nii(data_sub_dir, 'spmT_0002.nii', 'Z')
            zscore_nii(data_sub_dir, 'spmT_0003.nii', 'Z')
            print(sub,'was zscored.')
