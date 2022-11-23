import os
import time
import glob
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk


def run_fix_preprocess_parallel(preprocessed_dirs,train_weight,thr=20):
    start_time = time.time()
    fix_denoise_command = '/home/dell/NeuroApp/FIX/fix/fix {} {} {}'
    cmds_list = [fix_denoise_command.format(pdir,train_weight,thr) for pdir in preprocessed_dirs]
    procs_list = []

    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for pdir, proc in zip(preprocessed_dirs, procs_list):
        proc.wait()
        print("{} finished!".format(pdir))

    end_time = time.time()
    run_time = round((end_time - start_time) / 60 / 60, 2)
    print(f"Run time cost {run_time}")


if __name__ == "__main__":
    train_weight = '/home/dell/NeuroApp/FIX/fix/training_files/WhII_Standard.RData'
    thr = 10

    # filter subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')  # look out
    data = data.query("(game1_acc>=0.8)and(Age>=18)")
    subject_list = data['Participant_ID'].to_list()

    # subject_list = []
    #subject_list = ['sub-'+str(s).zfill(3) for s in subject_list]

    # Get all the paths!
    preprocessed_dir = r'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap'
    preprocessed_dirs_list = []
    for subj_id in subject_list:
        preprocessed_dirs_list.extend(glob.glob(opj(preprocessed_dir,
                                       f'{subj_id}/fsl_smooth0/'
                                       f'{subj_id}_task-game1_run-*_space-T1w_desc-preproc_bold.ica')))
    preprocessed_dirs_list.sort()

    preprocessed_dirs_chunk = list_to_chunk(preprocessed_dirs_list,24)
    for preprocessed_dirs in preprocessed_dirs_chunk:
        run_fix_preprocess_parallel(preprocessed_dirs,train_weight,thr)