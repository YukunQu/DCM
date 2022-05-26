# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import time
import pandas as pd
import subprocess


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
pid = data['Participant_ID'].to_list()
subjects = [p.split('_')[-1] for p in pid]
#%%
sub_list = []
sub_set_num = 0
sub_set = ''
for i,sub in enumerate(subjects):
    sub_set = sub_set+ sub + ' '
    sub_set_num = sub_set_num+1
    if sub_set_num == 5:
        sub_list.append(sub_set[:-1])
        sub_set_num = 0
        sub_set = ''
    elif i == (len(subjects)-1):
        sub_list.append(sub_set[:-1])
    else:
        continue
#%%
command_surfer = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --use-aroma --output-spaces MNI152NLin2009cAsym:res-2 T1w fsnative --no-tty -w {} --nthreads 20'
command_volume = 'fmriprep-docker {} {} participant --participant-label {} --fs-license-file {} --use-aroma --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym:res-native T1w --no-tty -w {} --fs-no-reconall --nthreads 66'

sub_list = ['015 027 037 046 076 080']
#
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
