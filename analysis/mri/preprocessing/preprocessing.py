# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import time
import subprocess


subjects = ['070 071 072 073']

command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w fsnative --no-tty -w {} --nthreads 20'
command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 66'
command_vnfmap = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --ignore fieldmaps --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 66'


starttime = time.time()
for subj in subjects:
    bids_dir = r'/mnt/workdir/DCM/BIDS'
    out_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume'
    
    work_dir = r'/mnt/workdir/DCM/working'
    freesurfer_license = r'/mnt/data/license.txt'
    command = command_volume.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2),'h')