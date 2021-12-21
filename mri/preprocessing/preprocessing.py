# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import subprocess

subjects = [14,15,16,17,18,21,22]
for subj in subjects:
    subj = str(subj).zfill(3)
    ori_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/NeuroData/MRI'.format(subj) 
    config = r'/mnt/data/Project/DCM/Config/config_sub{}.json'.format(subj)
    out_dir = r'/mnt/data/Project/DCM/BIDS'
    command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir,subj,config,out_dir)
    print("Command:",command)
    subprocess.call(command,shell=True)
    
#%%

import subprocess

subj = '016'
bids_dir = r'/mnt/data/Project/DCM/BIDS'
out_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/fmriprep'
freesurfer_license = r'/mnt/data/license.txt'
command = r'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 --fs-no-reconall --no-tty'.format(bids_dir,out_dir,subj,freesurfer_license)
print("Command:",command)
subprocess.call(command,shell=True)