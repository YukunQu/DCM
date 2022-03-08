# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:04:09 2021

@author: QYK
"""
import os
import pandas as pd


def reconstruct_map(blockdir,dim):
    
    # concat block file
    blockfiles = []
    for i in range(1,21):
        blockpath = os.path.join(blockdir, '{}_Block{}.xlsx'.format(dim,i))
        blockfiles.append(pd.read_excel(blockpath))
    trainCondition = pd.concat(blockfiles,axis=0)
    return trainCondition 


blockdir = r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2\train'
dim = 'ap'
trainCondition = reconstruct_map(blockdir,dim)
new_pairs_df = pd.read_excel(r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2_recon/ap_train.xlsx')