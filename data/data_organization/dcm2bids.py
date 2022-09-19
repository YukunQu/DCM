#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:02:34 2022

@author: dell
"""
#%%
import subprocess
def dcm2bids_helper(subjects):
    for subj in subjects:
        subj = str(subj).zfill(3)

        ori_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/workdir/DCM/tmp/{}_helper'.format(subj)
        command = r'dcm2bids_helper -d {} -o {}'.format(ori_dir,out_dir)
        print("Command:",command)
        subprocess.call(command,shell=True)

subjects = [167]
dcm2bids_helper(subjects)
#%%
import subprocess
def dcm2bids(subjects,config_file):
    for subj in subjects:
        subj = str(subj).zfill(3)
        config = config_file
        
        ori_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/workdir/DCM/BIDS'
        command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir,subj,config,out_dir)
        print("Command:",command)
        subprocess.call(command,shell=True)


# Peking scaning
config_pk = r'/mnt/workdir/DCM/config/config_Peking.json'

# CS scaning
config_ibp = r'/mnt/workdir/DCM/config/config_CS.json'

# dwi
config_dwi = r'/mnt/workdir/DCM/config/config_dwi.json'

# subject
individual_config = r'/mnt/workdir/DCM/config/config_sub167.json'

subjects_list = [167]
dcm2bids(subjects_list, individual_config)
#%%
# check the BIDS

import os
import pandas as pd

# read the participants.tsv file and got the subject list
df = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv',sep='\t')
subject_list = df.query("game1_fmri==1")['Participant_ID']

# check the existence of sourcedata for subjects
sourcedata_dir = r'/mnt/workdir/DCM/sourcedata'
for sub in subject_list:
    sub_mri_dir = os.path.join(sourcedata_dir,sub,'NeuroData','MRI')
    file_num = len(os.listdir(sub_mri_dir))
    if file_num == 0:
        print("The",sub,"have no MRI file!")

# check the BIDS for subjects
BIDS_dir = r'/mnt/workdir/DCM/BIDS'
sub_bids = os.listdir(BIDS_dir)
subject_list = [sub.replace("_",'-') for sub in subject_list]

for sub in subject_list:
    if sub not in sub_bids:
        print(sub)

sub_bids.sort()
for sub in sub_bids:
    if sub not in subject_list:
        print(sub)
print("The check finished.")