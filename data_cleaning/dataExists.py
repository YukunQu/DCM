#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# @author: qyk


#The scripts to check data
#%%
import os
# 检测fmri&MEG文件是否有缺损
data_dir = '/mnt/data/Project/DCM/BIDS/sourcedata'
subpath = r'Neuron/MEG'
file_size = 10240

subjects= os.listdir(data_dir)
subjects.sort()
for sub in subjects[:50]:
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
import os
# 检测哪个被试的某个文件夹下为空
def check_subj_data(data_dir,ranges,filters=None,stand_form=True):

    subjects= os.listdir(data_dir)
    subjects.sort()
    
    for sub in subjects[ranges[0]:ranges[1]]:
        print('-----------------{} data Index----------------'.format(sub))
        sub_data_dir = os.path.join(data_dir,sub)
        if stand_form:
            mod_list = os.listdir(sub_data_dir)
            for mod in mod_list:
                data_list = os.listdir(os.path.join(sub_data_dir,mod))
                for ddir in data_list:
                    if isinstance(filters,list):
                        if ddir in filters:
                            ddir_path = os.path.join(sub_data_dir,mod,ddir)
                            data_num = len(os.listdir(ddir_path))
                            print('File number of ',ddir,":",data_num)
                    else:
                        ddir_path = os.path.join(sub_data_dir,mod,ddir)
                        data_num = len(os.listdir(ddir_path))
                        print('File number of ',ddir,":",data_num)
        else:
            data_list = os.listdir(sub_data_dir)
            for ddir in data_list:
                ddir_path = os.path.join(sub_data_dir, ddir)
                data_num = len(os.listdir(ddir_path))
                print('File number of ',ddir,":",data_num)
        print("    ")
            

data_dir = '/mnt/data/Project/DCM/sourcedata'           
check_subj_data(data_dir, (48,64),['MEG','MRI','meg_task-1DInfer','fmri_task-game1',
                                   'fmri_task-game2-train','fmri_task-game2-test'])




