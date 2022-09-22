# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import time
import subprocess


fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_ica'
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
# split the subjects into subject units. Each unit includes only five subjects to prevent memory overflow.
sub_list = []
sub_set_num = 0
sub_set = ''
for i,sub in enumerate(unexist_subjects):
    sub_set = sub_set+ sub + ' '
    sub_set_num = sub_set_num+1
    if sub_set_num == 6:
        sub_list.append(sub_set[:-1])
        sub_set_num = 0
        sub_set = ''
    elif i == (len(unexist_subjects)-1):
        sub_list.append(sub_set[:-1])
    else:
        continue
#%%
command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --use-aroma --output-spaces MNI152NLin2009cAsym:res-2 T1w fsnative --no-tty -w {} --nthreads 20'
command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --use-aroma --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 66 '
# --ignore fieldmaps
sub_list = ['089 099 100 101 103 111']

starttime = time.time()
for subj in sub_list:
    bids_dir = r'/mnt/workdir/DCM/BIDS'
    out_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_ica'
    
    work_dir = r'/mnt/workdir/DCM/working'
    freesurfer_license = r'/mnt/data/license.txt'
    command = command_volume.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2), 'h')
