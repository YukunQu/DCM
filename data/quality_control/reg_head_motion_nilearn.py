import os
import pandas as pd
import glob
from joblib import Parallel, delayed
from nilearn.image import clean_img,smooth_img
from os.path import join as opj


def clean_image_and_save(func_img, motion_file, output_dir):
    nuisance_reg = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                    'csf', 'white_matter']
    motion = pd.read_csv(motion_file, sep="\t")
    motion = motion[nuisance_reg]
    motion = motion.fillna(0.0)

    cleaned_img = clean_img(func_img, confounds=motion, standardize=False, detrend=False, high_pass=0.01, t_r=3.0)
    cleaned_img = smooth_img(func_img, fwhm=8)
    output_file = os.path.join(output_dir, 'regfilt.nii.gz')
    os.makedirs(output_dir, exist_ok=True)
    cleaned_img.to_filename(output_file)


if __name__ == "__main__":
    # filter subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subject_list = data['Participant_ID'].to_list()[:2]

    # Get all the paths!
    func_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
    output_dir = r'/mnt/workdir/DCM/BIDS/derivatives/head_motion/preprocessed_data'
    func_imgs_list = []
    for subj_id in subject_list:
        func_imgs_list.extend(glob.glob(opj(func_dir,
                                            f'{subj_id}/func/{subj_id}_task-game1_run-*_space-T1w_desc-preproc_bold_trimmed.nii.gz')))
    func_imgs_list.sort()
    motion_files_list = [fp.replace('_space-T1w_desc-preproc_bold_trimmed.nii.gz',
                                    '_desc-confounds_timeseries_trimmed.tsv') for fp in func_imgs_list]

    output_dirs = [func_img.replace(func_dir, output_dir).replace('bold_trimmed.nii.gz','bold_trimmed') for func_img in func_imgs_list]

    # Run the parallel job
    Parallel(n_jobs=20)(delayed(clean_image_and_save)(func_img, motion_file, output_dir)
                       for func_img, motion_file, output_dir in zip(func_imgs_list, motion_files_list, output_dirs))
