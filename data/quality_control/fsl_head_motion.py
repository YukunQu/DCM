# batch script to calculate the mean FD using FSL command

import os
import time
import glob
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk

def run_fsl_motion_outliers_parallel(func_imgs,output_dirs):
    start_time = time.time()

    fsl_moutliers_command = 'fsl_motion_outliers -i {} -o {} -s {} -p {} --fdrms'

    cmd_list = []
    for func_img,output_dir in zip(func_imgs,output_dirs):
        output_confound = os.path.join(output_dir, 'confound.txt')
        output_metrics = os.path.join(output_dir, 'metrics.txt')
        output_plot = os.path.join(output_dir, 'plot.png')
        os.makedirs(output_dir,exist_ok=True)
        cmd_list.append(fsl_moutliers_command.format(func_img, output_confound, output_metrics, output_plot))

    procs_list = []
    for cmd in cmd_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for odir, proc in zip(output_dirs, procs_list):
        proc.wait()
    print("{} finished!".format(odir))

    end_time = time.time()
    run_time = round((end_time - start_time) / 60 / 60, 2)
    print(f"Run time cost {run_time}")



if __name__ == "__main__":
    # filter subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subject_list = data['Participant_ID'].to_list()

    # Get all the paths!
    func_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
    output_dir = r'/mnt/workdir/DCM/BIDS/derivatives/head_motion/preprocessed_data'
    func_imgs_list = []
    for subj_id in subject_list:
        # func_imgs_list.extend(glob.glob(opj(func_dir,
        #                                             f'{subj_id}/func/{subj_id}_task-*_run-*_bold.nii.gz')))
        func_imgs_list.extend(glob.glob(opj(func_dir,
                                            f'{subj_id}/func/{subj_id}_task-*_run-*_space-T1w_desc-preproc_bold_trimmed.nii.gz')))
    func_imgs_list.sort()

    func_imgs_chunk = list_to_chunk(func_imgs_list,50)
    for func_imgs in func_imgs_chunk:
        output_dirs = [func_img.replace(func_dir,output_dir) for func_img in func_imgs]
        output_dirs = [output_dir.replace('bold_trimmed.nii.gz','bold_trimmed') for output_dir in output_dirs]
        run_fsl_motion_outliers_parallel(func_imgs,output_dirs)