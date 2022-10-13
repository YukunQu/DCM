# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import os
import time
import subprocess

import pandas as pd
#%%
"""
fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume'
exist_subjects = []
for file in os.listdir(fmriprep_dir):
    if 'sub-' in file:
        exist_subjects.append(file)

bids_dir = r'/mnt/workdir/DCM/BIDS'
unexist_subjects = []
for file in os.listdir(bids_dir):
    if 'sub-' in file:
        if file not in exist_subjects:
            unexist_subjects.append(file)  # filter the subjects who are already exist

unexist_subjects.sort()
#%%
bids_dir = r'/mnt/workdir/DCM/BIDS'
subject_list = os.listdir(bids_dir)
subject_list = [sub for sub in subject_list if 'sub-' in sub]
"""
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
pid = data['Participant_ID'].to_list()

#target_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/separate_hexagon/Setall/6fold'
#already_sub = os.listdir(target_dir)
#subject_list = [p.split('-')[-1] for p in pid if p not in already_sub]
subject_list = [p.split('-')[-1] for p in pid]
subject_list.sort()
subject_list = subject_list[-68:]

# split the subjects into subject units. Each unit includes only five subjects to prevent memory overflow.
sub_list = []
sub_set_num = 0
sub_set = ''
for i,sub in enumerate(subject_list):
    sub_set = sub_set+ sub + ' '
    sub_set_num = sub_set_num+1
    if sub_set_num == 12:
        sub_list.append(sub_set[:-1])
        sub_set_num = 0
        sub_set = ''
    elif i == (len(subject_list)-1):
        sub_list.append(sub_set[:-1])
    else:
        continue


#%%
#command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w --no-tty -w {} --use-syn-sdc --nthreads 100'
command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w --no-tty -w {} --use-syn-sdc --nthreads 60 --fs-no-reconall'

sub_list = ['113 114 115 116 117 118 119']
starttime = time.time()
for subj in sub_list:
    bids_dir = r'/mnt/workdir/DCM/BIDS'
    out_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume'
    
    work_dir = r'/mnt/workdir/DCM/working'
    freesurfer_license = r'/mnt/data/license.txt'
    command = command_volume.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2), 'h')