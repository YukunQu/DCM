#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:21:08 2022

@author: dell
"""

# 
import os
import numpy as np
from nilearn import image


def zscore_img(img_path):
    img  = image.load_img(img_path)
    img_zscore = image.math_img("(img - img.mean())/img.std()",img=img)
    return img_zscore


data_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Nipype/hexonDe_result/1stLevel'
sub_list = os.listdir(data_dir)
for sub in sub_list:
    img_path = os.path.join(data_dir,sub,'ess_0004.nii')
    img = image.load_img(img_path)
    img_zscore = zscore_img(img)
    save_name = os.path.join(data_dir,sub,'ess_0004_Z.nii')
    img_zscore.to_filename(save_name)