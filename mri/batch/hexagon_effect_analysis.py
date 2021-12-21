#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:42:40 2021

@author: dell
"""

import os   
from os.path import join
from mri.analysis.event import hexagon_effect_onM2_event,testFai_effect_event,hexagon_effect_onM2_event_spec
from mri.analysis.estimate_fai import estimate_fai
from mri.analysis.estimate_hexagon_effect import estimate_hexagon_effect

subjects = ['017','018','021','022']
train_run =  [1,2,3,4]
test_run = [5,6]
mask_path = r'/mnt/data/Template/EC_anatomy_roi.nii.gz'


for subid in subjects:
    hexagon_effect_onM2_event_spec(subid)
    # make result data structure
    save_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/nilearn/cv1'
    save_dir = join(save_dir,'sub-{}'.format(subid), 'game1')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ifold_fai = estimate_fai(subid, train_run, mask_path, save_dir)
     
    # test beta     
    testFai_effect_event(subid,test_run,ifold_fai)
    estimate_hexagon_effect(subid, test_run, mask_path, save_dir)