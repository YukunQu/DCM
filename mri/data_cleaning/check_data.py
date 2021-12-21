#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# @author: qyk


#The scripts to check data
#%%
import os
# 检测fmri&MEG文件是否有缺损
data_dir = '/mnt/data/Project/DCM/BIDS/sourcedata'
subpath = r'Behaviour/meg_task-1DInfer'
file_size = 10240

subjects= os.listdir(data_dir)
subjects.sort()
for sub in subjects[:30]:
    fmri_data_dir = os.path.join(data_dir,sub,subpath)
    fmri_tmp_list = os.listdir(fmri_data_dir)
    file_num = len(fmri_tmp_list)
    print('File number of ',sub,":",file_num)
    for fmri_data in fmri_tmp_list:
        size_tmp = os.path.getsize(os.path.join(fmri_data_dir,fmri_data))
        if size_tmp < file_size:
            print(sub,fmri_data,"不通过")
    print("{}检查完成".format(sub))

#%%
# 检测哪个被试的某个文件夹下为空
data_dir = '/mnt/data/Project/DCM/BIDS/sourcedata'
#subpath = r'Behaviour/fmri_task-game2-train'
subpath = r'Behaviour/train_dim1'

subjects= os.listdir(data_dir)
subjects.sort()
for sub in subjects[:27]:
    sub_data_dir = os.path.join(data_dir,sub,subpath)
    data_num = len(os.listdir(sub_data_dir))
    print('File number of ',sub,":",data_num)