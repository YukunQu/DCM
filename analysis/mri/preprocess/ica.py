import os
import time
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
import glob

def run_melodic_ica_parallel(func_list):
    start_time = time.time()
    melodic_decomposition_command = 'melodic -i {} --tr=3 --mmthresh=0.5 --report -v'
    cmds_list = [melodic_decomposition_command.format(func_data) for func_data in func_list]
    procs_list = []
    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for outname, proc in zip(func_list, procs_list):
        proc.wait()
        print("{} finished!".format(outname))

    end_time = time.time()
    run_time = round((end_time - start_time) / 60 / 60, 2)
    print(f"Run time cost {run_time}")


# filter subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
hp_data = data.query("(game1_acc>=0.8)and(Age>=18)")
subject_list = hp_data['Participant_ID'].to_list()


# split the subjects into many subject chunks. Each chunk includes only five subjects to prevent memory overflow.
sub_list = []
sub_set_num = 0
sub_set = []
for i, sub in enumerate(subject_list):
    sub_set.append(sub)
    if len(sub_set) == 5:
        sub_list.append(sub_set)
        sub_set = []
    elif i == (len(subject_list) - 1):
        sub_list.append(sub_set)
    else:
        continue

fmriprep_dir = '/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap'
for subject_chunk in sub_list:
    func_list = []
    for subj_id in subject_chunk:
        sub_dir = f'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap/{subj_id}'
        func_list.extend(glob.glob(opj(sub_dir,f'fsl_smooth0/{subj_id}_task-game1_run-*_space-T1w_desc-preproc_bold.ica/filtered_func_data.nii.gz')))
    func_list.sort()
    run_melodic_ica_parallel(func_list)