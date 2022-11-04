import os
import time
import pandas as pd
import subprocess

fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/mriqc'
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
    sub_set = sub_set + sub + ' '
    sub_set_num = sub_set_num+1
    if sub_set_num == 8:
        sub_list.append(sub_set[:-1])
        sub_set_num = 0
        sub_set = ''
    elif i == (len(unexist_subjects)-1):
        sub_list.append(sub_set[:-1])
    else:
        continue

#%%
mriqc_participant = 'docker run --rm -v {}:/data:ro -v {}:/out nipreps/mriqc:latest /data /out participant --participant_label {} -m T1w bold -v --omp-nthreads 66'
mriqc_group = 'docker run --rm -v {}:/data:ro -v {}:/out nipreps/mriqc:latest /data /out group -m T1w bold -v --omp-nthreads 66'

bids_dir = r'/mnt/workdir/DCM/BIDS'
out_dir = r'/mnt/workdir/DCM/BIDS/derivatives/mriqc'

starttime = time.time()

for subj in sub_list:
    command = mriqc_participant.format(bids_dir,out_dir,subj)
    print("Command:",command)
    subprocess.call(command, shell=True)

command = mriqc_group.format(bids_dir,out_dir)
print("Command:",command)
subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2), 'h')