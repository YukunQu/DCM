#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# @author: qyk

# The scripts check data for each subject

import os
# 检测哪个被试的某个文件夹下为空

def check_subj_data(data_dir,sub_list,filters=None,stand_form=True):
    with open(os.path.join(data_dir, 'data_exist_state.txt'),'w') as f:
        for sub in sub_list:
            print('-----------------{} data Index----------------'.format(sub))
            f.write('-----------------{} data Index----------------'.format(sub))
            f.write('\n')
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
                                print('File number of ', ddir,":",data_num)
                                f.write(f'File number of {ddir}: {data_num}')
                                f.write('\n')
                        else:
                            ddir_path = os.path.join(sub_data_dir,mod,ddir)
                            data_num = len(os.listdir(ddir_path))
                            print('File number of ', ddir,":",data_num)
                            f.write(f'File number of {ddir}: {data_num}')
                            f.write('\n')
            else:
                data_list = os.listdir(sub_data_dir)
                for ddir in data_list:
                    ddir_path = os.path.join(sub_data_dir, ddir)
                    data_num = len(os.listdir(ddir_path))
                    print('File number of ',ddir,":",data_num)
                    f.write(f'File number of {ddir}: {data_num}')
                    f.write('\n')
            f.write('\n')


#data_dir = '/mnt/data/Sourcedata/DCM'
data_dir = '/mnt/workdir/DCM/sourcedata'
sub_list = ['sub_'+str(i).zfill(3) for i in range(95,150)]
check_subj_data(data_dir, sub_list,['MEG','MRI','mixed_test','meg_task-1DInfer',
                                    'pilot','fmri_task-game1','fmri_task-game2-train',
                                    'fmri_task-game2-test','placement'])