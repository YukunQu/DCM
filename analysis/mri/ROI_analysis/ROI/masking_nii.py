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


def masking_nii(source_dir,file,prefix):
    filepath = os.path.join(source_dir,file)
    mask = load_img(r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')
    fmap = load_img(filepath)

    fmap_data = fmap.get_fdata()
    mask_data = mask.get_fdata()
    fmap_data[mask_data == 0] = np.float64('NaN')
    map_masked = new_img_like(fmap, fmap_data)
    map_masked.to_filename(os.path.join(source_dir,prefix+file))

#%%
task = 'game2'
glm_type = 'separate_hexagon'

for set_id in ['all']:
    testset_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/{}/Set{}'.format(task,glm_type,set_id)

    for ifold in range(6,7):
        ifold = f'{ifold}fold'
        data_dir = os.path.join(testset_dir,ifold)
        sub_list = os.listdir(data_dir)
        for sub in sub_list:
            data_sub_dir = os.path.join(data_dir,sub)
            masking_nii(data_sub_dir, 'con_0009.nii', 'M')
            masking_nii(data_sub_dir, 'con_0010.nii', 'M')
            print(sub,'was masked.')

#%%
task = 'game1'
glm_type = 'separate_hexagon'

for set_id in [1,2]:
    cmap_dir= r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/{}/Set{}'.format(task,glm_type,set_id)

    for ifold in range(4,9):
        ifold = f'{ifold}fold'
        data_dir = os.path.join(cmap_dir,ifold)
        sub_list = os.listdir(data_dir)
        for sub in sub_list:
            data_sub_dir = os.path.join(data_dir,sub)
            masking_nii(data_sub_dir, 'con_0009.nii', 'M')
            masking_nii(data_sub_dir, 'con_0010.nii', 'M')
            #masking_nii(data_sub_dir, 'con_0003.nii', 'M')
            #masking_nii(data_sub_dir, 'con_0004.nii', 'M')
            print(sub,'was masked.')

#%%
testset_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/alignPhi/EC_park/Set{}'

for set_id in [1,2]:
    for ifold in range(4,9):
        ifold = f'{ifold}fold'
        data_dir = os.path.join(testset_dir.format(set_id),ifold)
        sub_list = os.listdir(data_dir)
        for sub in sub_list:
            data_sub_dir = os.path.join(data_dir,sub)
            masking_nii(data_sub_dir, 'spmT_0001.nii', 'M')
            masking_nii(data_sub_dir, 'spmT_0002.nii', 'M')
            masking_nii(data_sub_dir, 'spmT_0003.nii', 'M')
            print(sub,'was zscored.')