#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:23:00 2021

@author: dell
"""

import os
import shutil
import pandas as pd 
#%%
participant_file_path = r'/mnt/data/Project/DCM/participants_exp.xlsx'
subject_df = pd.read_excel(participant_file_path)

data_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata'
subjects = os.listdir(data_dir)
subjects.sort()

origin_file_dir = r'/mnt/data/Project/DCM/tmp/2022.1.17train_data/total_test' # flexible change
origin_file_list = os.listdir(origin_file_dir)

copy_count = 0
have_count = 0
for sub in subjects[27:50]:
    subinfo = subject_df[subject_df['Participant_ID']==sub]
    try:
        exp_id = subinfo['Exp_ID'].values[0]
    except:
        print(sub,"don't have file. ")
        continue
    for ori_file in origin_file_list:
        file_id = ori_file.split('_')[0]
        if (exp_id.upper() == file_id) or (exp_id.lower() == file_id):
            ori_file_path = os.path.join(origin_file_dir,ori_file)
            tar_path = os.path.join(data_dir,sub,'Behaviour/total_test',ori_file)# flexible change
            if os.path.exists(tar_path):
                print(sub,'—',ori_file,'already have this file.')
                have_count +=1
            else:
                shutil.copy(ori_file_path, tar_path)
                copy_count +=1
                print(sub,ori_file,'copy completed.')

print("Total Copyed file:",copy_count)
print("Total already existed file:",have_count)

#%%
# 将符合标准的MEG行为数据移动到BIDS/derivatives/behaviour/data 目录下
import os 
from os.path import join
import shutil

source_dir = r'/mnt/workdir/DCM/sourcedata'
subjects = os.listdir(source_dir)
subjects.sort()

analysis_data_dir = r'/mnt/workdir/DCM/BIDS/derivatives/behaviour/data'

for subj in subjects:
    ori_meg_datadir = join(source_dir,subj,r'Behaviour','meg_task-1DInfer')
    tar_meg_datadir = join(analysis_data_dir,subj,'meg_task-1DInfer')
    meg_tmp_list = os.listdir(ori_meg_datadir)
    meg_data_list = []
    for file in meg_tmp_list:
        file_size = os.path.getsize(join(ori_meg_datadir,file))
        if ('.csv' in file) and (file_size>12288):
            if ('loop' not in file) and ('trial' not in file):
                meg_data_list.append(file)
    mv_file_num = 0                
    for file in meg_data_list:
        file_path = join(ori_meg_datadir,file)
        tar_path = join(tar_meg_datadir,file)
        if not os.path.exists(tar_path):
            shutil.copy(file_path,tar_path)
            mv_file_num += 1
    print(subj,"copy completed.",'Copy file:',mv_file_num)