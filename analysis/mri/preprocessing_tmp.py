# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import time
import pandas as pd
import subprocess

command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --use-aroma --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 15'

#sub_list = ['084 085 090 091 092', '096 097 098 100 102']
sub_list = ['096 097 098 100 102']
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
print('总共的时间为:', round((endtime - starttime)/60/60,2),'h')