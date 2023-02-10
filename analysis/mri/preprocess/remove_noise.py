import os
import glob
import time
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
from collections import deque


def tail(filename, n=10):
    """Return the last n lines of a file"""
    return deque(open(filename), n)


def get_filter(filename, n=1):
    """Get component from the labels file"""
    filters = tail(filename, 1)[0]
    for osign in ['[', ']', ' ', "\n"]:
        filters = filters.replace(osign, '')
    return filters


def remove_noise(input_list, dlist, output_list, filter_list):
    """remove noise from functional data in parallel"""
    fsl_regfilt_command = 'fsl_regfilt -i {} -d {} -o {} -f {}'

    cmds_list = [fsl_regfilt_command.format(inp, d, o, f)
                 for inp, d, o, f in zip(input_list, dlist, output_list, filter_list)]

    procs_list = []
    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for outname, proc in zip(output_list, procs_list):
        proc.wait()
        print("{} finished!".format(outname))


if __name__ == "__main__":
    # sepcify subjects
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
        if len(sub_set) == 4:
            sub_list.append(sub_set)
            sub_set = []
        elif i == (len(subject_list) - 1):
            sub_list.append(sub_set)
        else:
            continue

    start_time = time.time()
    fmriprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep'
    for subject_chunk in sub_list:
        # prepare parameter list
        input_list = []
        filter_list = []
        for subj_id in subject_chunk:
            input_list.extend(glob.glob(opj(fmriprep_dir,
                                            f'{subj_id}/func/{subj_id}_task-game1_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_blod_smooth8.nii')))
        input_list.sort()

        dlist = [f.replace('.nii', '.ica/melodic_mix') for f in input_list]
        output_list = [f.replace('blod_smooth8.nii', 'bold_smooth8_cleaned.nii') for f in input_list]

        for f in input_list:
            ica_dir = f.replace('.nii','.ica')
            labels = opj(ica_dir,'labels.txt')
            filter_list.append(get_filter(labels, 1))

        remove_noise(input_list, dlist, output_list, filter_list)  # remove noise

    end_time = time.time()
    run_time = round((end_time - start_time) / 60 / 60, 2)
    print(f"Run time cost {run_time}")
