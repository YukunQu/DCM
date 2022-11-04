# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import os
import time
import subprocess

import pandas as pd
#%%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
pid = data['Participant_ID'].to_list()

fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume'
exist_subjects = []
for file in os.listdir(fmriprep_dir):
    if 'sub-' in file:
        exist_subjects.append(file)

unexist_subjects = []
for f in pid:
    if f not in exist_subjects:
        unexist_subjects.append(f)  # filter the subjects who are already exist
#%%
subject_list = [p.split('-')[-1] for p in unexist_subjects]
subject_list.sort()

# split the subjects into subject units. Each unit includes only five subjects to prevent memory overflow.
sub_list = []
sub_set_num = 0
sub_set = ''
for i,sub in enumerate(subject_list):
    sub_set = sub_set+ sub + ' '
    sub_set_num = sub_set_num+1
    if sub_set_num == 10:
        sub_list.append(sub_set[:-1])
        sub_set_num = 0
        sub_set = ''
    elif i == (len(subject_list)-1):
        sub_list.append(sub_set[:-1])
    else:
        continue
#%%
#command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w --no-tty -w {} --use-syn-sdc --nthreads 100'
#command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w --no-tty -w {} --use-syn-sdc --nthreads 80 --fs-no-reconall'
command_volume_ignore_fmap = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w --no-tty -w {} --nthreads 32 --fs-no-reconall --ignore fieldmaps --use-syn-sdc'

starttime = time.time()
for subj in sub_list:
    bids_dir = r'/mnt/workdir/DCM/BIDS'
    out_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume'
    
    work_dir = r'/mnt/workdir/DCM/working'
    freesurfer_license = r'/mnt/data/license.txt'
    command = command_volume_ignore_fmap.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2), 'h')