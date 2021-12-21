#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:49:21 2021

@author: quyukun
"""
import os 
import pandas as pd 
import nibabel
import numpy as np


prepath = r'/nfs/s2/userhome/quyukun/workingdir/fmriprep/data/bold/tsnr/sub-{}/ses-{}'
describe_data = pd.DataFrame(columns=['name','mean','std','max'])
for sub in ['01','02']:
    for ses in ['1']:
        file_dir = prepath.format(sub,ses)
        file = os.listdir(file_dir)
        tsnr_file = [f  for f in file if 'tsnr' in f]
        for tsnr_map in tsnr_file:
            tsnr_filepath = os.path.join(file_dir,tsnr_map)
            data = nibabel.load(tsnr_filepath).get_data()
            mean = data.mean()
            std = data.std()
            data_row = pd.DataFrame({'name':tsnr_map,'mean':[mean],'std':[std],'max':[data.max()]})
            describe_data = describe_data.append(data_row)
describe_data.to_excel(r'/nfs/s2/userhome/quyukun/workingdir/fmriprep/data/bold/tsnr/tsnr_describe.xlsx',index='False')