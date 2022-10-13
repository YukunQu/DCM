#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# @author: qyk

# The scripts check data for each subject

import os

#%%
# 检测原始文件夹下被试数据是否完整
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
                    if  mod not in ['Behaviour','NeuroData']:
                        continue
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


data_dir = '/mnt/workdir/DCM/sourcedata'
sub_list = ['sub_'+str(i).zfill(3) for i in range(191,196)]
check_subj_data(data_dir, sub_list,['MEG','MRI','mixed_test','meg_task-1DInfer',
                                    'pilot','fmri_task-game1','fmri_task-game2-train',
                                    'fmri_task-game2-test','placement'])

#%%
# 检测BIDS 目录下被试数据是否完整

def check_bids_dir():
    import os
    import pandas as pd
    # read the participants.tsv file and got the subject list
    df = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv',sep='\t')
    subject_list = df.query("game1_fmri ==1")['Participant_ID']
    subject_list = [sub.replace("_", '-') for sub in subject_list]

    # check the BIDS for subjects
    BIDS_dir = r'/mnt/workdir/DCM/BIDS'
    sub_bids = os.listdir(BIDS_dir)

    for sub in subject_list:
        if sub not in sub_bids:
            print(sub,"shoud have BIDS files but not.")

    sub_bids.sort()
    for sub in sub_bids:
        if 'sub' in sub:
            if sub not in subject_list:
                print(sub,"doesn't need the BIDS files.")
    print("The BIDS checking finished.")

check_bids_dir()
#%%

def check_fmriprep_dir():
    import os
    import pandas as pd
    # read the participants.tsv file and got the subject list
    df = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv',sep='\t')
    subject_list = df.query("game1_fmri==1")['Participant_ID']

    # check the BIDS for subjects
    fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_surfer/fmriprep'
    freesurfer_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_surfer/freesurfer'

    for dir in [fmriprep_dir, freesurfer_dir]:
        print("------------------{}------------------".format(dir.split('/')[-1]))
        sub_preprocess_list = os.listdir(dir)
        subject_list = [sub.replace("_",'-') for sub in subject_list]
        subject_list.sort()
        i = 0
        for sub in subject_list:
            if sub not in sub_preprocess_list:
                i +=1
                print(sub,"should have preprocessing files but not.")
        print(i,"subjects have no-preprocessing files")

        sub_preprocess_list.sort()
        for sub in sub_preprocess_list:
            if ('sub' in sub) and ('html' not in sub):
                if sub not in subject_list:
                    print(sub,"doesn't need the preprocessing files.")
        print("The check finished.")
check_fmriprep_dir()