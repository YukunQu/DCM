# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import time
import subprocess

#subjects = ['005 006 010 011 012 015 016 017','018 022 023 024 027 029 030 031',
#            '032 033 036 037 038 043 046 047','048 049 051 052 053']
#'010 011 012 015 023 040 048 049'
subjects = ['050 055 059 060 061 062 063 064']

command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 T1w fsnative --no-tty -w {} --nthreads 18'
command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 66'
command_vnfmap = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --ignore fieldmaps --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 66'


starttime = time.time()
for subj in subjects:
    bids_dir = r'/mnt/data/Project/DCM/BIDS'
    out_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/fmriprep_volume'
    
    work_dir = r'/mnt/data/Project/DCM/working'
    freesurfer_license = r'/mnt/data/license.txt'
    command = command_volume.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2),'h')