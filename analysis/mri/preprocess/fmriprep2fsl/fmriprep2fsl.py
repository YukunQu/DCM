import os
import shutil
import glob
import numpy as np
import pandas as pd
from subprocess import call
from nipype.interfaces.fsl import maths
from nilearn.image import resample_to_img
from analysis.mri.img.mask_img import mask_img
from joblib import Parallel, delayed
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk


def fmriprep2fsl(sub):
    """
    convert fmriprep files into a FSL format dictory for ICA and denoise
    :param sub:
    :param subd:a subject directory of fmripre format
    :param outd:output path of FSL format dictory
    """
    subd = rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/{sub}'
    outd = os.path.join(subd, 'fsl')
    if not os.path.exists(outd):
        os.mkdir(outd)
    sub_id = subd.split('/')[-1]
    func_template = os.path.join(subd, 'func', 'sub-*_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz')
    func_list = glob.glob(func_template)
    func_list.sort()
    for func_file in func_list:
        # create func directory
        func_name = func_file.split('/')[-1]
        print(func_name, 'start!')
        func_name = func_name.replace('.nii.gz', '.ica')
        out_func_dir = os.path.join(outd, func_name)
        if not os.path.exists(out_func_dir):
            os.mkdir(out_func_dir)

        # generate mean_func.nii.gz
        mean_func_outpath = os.path.join(out_func_dir, 'mean_func.nii.gz')
        mean_cmd = f'fslmaths {func_file} -Tmean {mean_func_outpath}'
        print("Command:", mean_cmd)
        call(mean_cmd, shell=True)

        # filter func image and generate to target directory
        out_func_path = os.path.join(out_func_dir, 'filtered_func_data.nii.gz')
        hpf_cmd = f'fslmaths {func_file} -bptf 100 -1 -add {mean_func_outpath} {out_func_path}'
        print("Command:", hpf_cmd)
        call(hpf_cmd, shell=True)

        # covert example_func.nii.gz
        example_func_file = func_file.replace('_desc-preproc_bold.nii.gz', '_boldref.nii.gz')
        reg_dir = os.path.join(out_func_dir, 'reg')
        if not os.path.exists(reg_dir):
            os.mkdir(reg_dir)
        out_example_func_path = os.path.join(reg_dir, 'example_func.nii.gz')
        shutil.copy(example_func_file,out_example_func_path)

        # covert T1w images and T1w mask
        mask_file = glob.glob(os.path.join(subd, 'anat', f'{sub_id}_desc-brain_mask.nii.gz'))[0]
        mask_outpath = os.path.join(out_func_dir, 'mask.nii.gz')
        resampled_mask = resample_to_img(mask_file, func_file, 'nearest')  # resample mask to func image
        resampled_mask.to_filename(mask_outpath)
        di = maths.DilateImage(in_file=mask_outpath,
                               operation="mean",
                               kernel_shape="sphere",
                               kernel_size=6,
                               out_file=mask_outpath)
        di.run()

        #
        anat_file = glob.glob(os.path.join(subd, 'anat', f'{sub_id}_desc-preproc_T1w.nii.gz'))[0]
        anat_outpath = os.path.join(out_func_dir, 'reg', 'highres.nii.gz')
        mask_img(anat_file, mask_file, anat_outpath)  # using fmriprep's brain mask to do skull stripping

        # motion parameters
        confound_file = func_file.replace('_space-T1w_desc-preproc_bold.nii.gz', '_desc-confounds_timeseries.tsv')
        motions_df = pd.read_csv(confound_file, sep='\t')
        motion_columns = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
        motions = motions_df[motion_columns].to_numpy()
        mc_dir = os.path.join(out_func_dir, 'mc')
        if not os.path.exists(mc_dir):
            os.mkdir(mc_dir)
        motion_outpath = os.path.join(mc_dir, 'prefiltered_func_data_mcf.par')
        np.savetxt(motion_outpath, motions, delimiter='  ')

        # create affine matrix
        mat_outpath = os.path.join(out_func_dir, 'reg', 'highres2example_func.mat')
        flirt_cmd = 'flirt -in {} -ref {} -omat {}'.format(anat_outpath, out_example_func_path, mat_outpath)
        print("Command:", flirt_cmd)
        call(flirt_cmd, shell=True)


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subject_list = data['Participant_ID'].to_list()
    subject_list = ['sub-130']
    subjects_chunk = list_to_chunk(subject_list,6)
    for chunk in subjects_chunk:
        results_list = Parallel(n_jobs=60)(delayed(fmriprep2fsl)(sub_id) for sub_id in chunk)