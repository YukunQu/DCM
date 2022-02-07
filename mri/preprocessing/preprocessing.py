# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import subprocess

# Peking scaning
subjects = [48,49,51,52,53]
for subj in subjects:
    subj = str(subj).zfill(3)
    ori_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/NeuroData/MRI'.format(subj) 
    config = r'/mnt/data/Project/DCM/Config/config_Peking.json'
    out_dir = r'/mnt/data/Project/DCM/BIDS'
    command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir,subj,config,out_dir)
    print("Command:",command)
    subprocess.call(command,shell=True)
    


#%%
# ibp scaning
subjects = [25,26,27,30,37,40,42]
for subj in subjects:
    subj = str(subj).zfill(3)
    ori_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/NeuroData/MRI'.format(subj) 
    config = r'/mnt/data/Project/DCM/Config/config_CS.json'
    out_dir = r'/mnt/data/Project/DCM/BIDS'
    command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir,subj,config,out_dir)
    print("Command:",command)
    subprocess.call(command,shell=True)
    


#%%
import time
import subprocess

command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w fsnative --no-tty -w {} --nthreads 18'
command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w --no-tty -w {} --fs-no-reconall --nthreads 28'

subjects = ['048 049 051 052 053']
starttime = time.time()
for subj in subjects:
    bids_dir = r'/mnt/data/Project/DCM/BIDS'
    out_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/fmriprep_surfer'
    work_dir = r'/mnt/data/Project/DCM/working'
    freesurfer_license = r'/mnt/data/license.txt'
    command = command_surfer.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2),'h')