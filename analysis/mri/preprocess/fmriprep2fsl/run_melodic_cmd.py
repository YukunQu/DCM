""" Using melodic command run ICA analysis for preprocessed functional data"""
import time
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
import glob


def run_melodic_ica_parallel(func_list):
    start_time = time.time()
    melodic_decomposition_command = 'melodic -i {} --tr=3 --report -v'
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


if __name__ == "__main__":
    # filter subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subject_list = data['Participant_ID'].to_list()
    subject_list.sort()
    # split the subjects into many subject chunks. Each chunk includes only five subjects to prevent memory overflow.
    sub_list = []
    sub_set_num = 0
    sub_set = []
    for i, sub in enumerate(subject_list):
        sub_set.append(sub)
        if len(sub_set) == 12:
            sub_list.append(sub_set)
            sub_set = []
        elif i == (len(subject_list) - 1):
            sub_list.append(sub_set)
        else:
            continue

    fmriprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
    func_template = '{}_task-*_run-*_space-T1w_desc-preproc_bold_trimmed.ica/filtered_func_data.nii.gz'
    for subject_chunk in sub_list:
        func_list = []
        for subj_id in subject_chunk:
            sub_dir = opj(fmriprep_dir,f'{subj_id}')
            func_list.extend(glob.glob(opj(sub_dir,'fsl',func_template.format(subj_id))))
        func_list.sort()
        run_melodic_ica_parallel(func_list)
