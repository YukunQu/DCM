# -*- coding: utf-8 -*-
"""
# distance effect
@author: QYK
"""
import os
import numpy as np
import pandas as pd
from os.path import join
from nilearn.image import math_img, load_img, resample_to_img, concat_imgs
from nilearn.masking import apply_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix

from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk

from joblib import Parallel, delayed, Memory

memory = Memory(cachedir='/tmp/joblib', verbose=0, bytes_limit=80 * 1024 ** 3)


def load_ev_rsa(event_path):
    event_condition = pd.read_csv(event_path, sep='\t')
    return event_condition


def prepare_data(subj, run_list, ifold, configs, concat_runs=False):
    """prepare images and design matrixs from different run """

    tr = configs['TR']
    func_dir = configs['func_dir']
    event_dir = configs['event_dir']

    task = configs['task']
    glm_type = configs['glm_type']

    func_name = configs['func_name']
    events_name = configs['events_name']
    regressor_file = configs['regressor_name']
    ifold = '{}fold'.format(ifold)

    functional_imgs = []
    design_matrices = []
    for i, run_id in enumerate(run_list):
        # load image
        func_path = join(func_dir, f'sub-{subj}', 'func', func_name.format(subj, run_id))
        func_img = load_img(func_path)
        functional_imgs.append(func_img)

        # load event
        event_path = join(event_dir, task, glm_type, f'sub-{subj}', ifold, events_name.format(subj, run_id))
        event = load_ev_rsa(event_path)

        # load motion
        add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                         'csf', 'white_matter']
        confound_file = os.path.join(func_dir, f'sub-{subj}', 'func', regressor_file.format(subj, run_id))
        confound_factors = pd.read_csv(confound_file, sep="\t")
        motion = confound_factors[add_reg_names]
        motion = motion.fillna(0.0)

        # creat design matrix
        n_scans = func_img.shape[-1]
        frame_times = np.arange(n_scans) * tr
        high_pass_fre = 1 / 100
        design_matrix = make_first_level_design_matrix(
            frame_times,
            event,
            hrf_model='spm',
            drift_model=None,
            high_pass=high_pass_fre, add_regs=motion, add_reg_names=add_reg_names, orth=False)
        design_matrices.append(design_matrix)

    # concatenate runs
    if concat_runs:
        functional_imgs = concat_imgs(functional_imgs)

        new_design_matrices = []
        for run_label, dm in enumerate(design_matrices):
            dm_columns = dm.columns
            new_dm_columns = ['s' + str(run_label + 1) + '_' + c for c in dm_columns]
            dm.columns = new_dm_columns
            new_design_matrices.append(dm)
        design_matrices = pd.concat(new_design_matrices, axis=0, ignore_index=True)
        design_matrices = design_matrices.fillna(0.0)
        print('1')
    return functional_imgs, design_matrices


def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def get_reg_index(design_matrix, target_name):
    target_index = []
    for i, reg_name in enumerate(design_matrix.columns):
        if target_name in reg_name:
            target_index.append(i)
    if len(target_index) == 0:
        print("The {} don't have regressor.".format(target_name))
    return target_index


def set_rsa_contrasts(design_matrix):
    conditions = design_matrix.columns
    conditions_names = list(set(conditions))
    conditions_names.sort()

    # base contrast
    contrasts_set = dict()
    for contrast_id in conditions_names:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector

    # advanced contrast
    contrasts_set['distance'] = contrasts_set['M2xdistance'] + contrasts_set['decisionxdistance']
    return contrasts_set


def first_level_glm(datasink, run_imgs, design_matrices):
    if not os.path.exists(datasink):
        os.makedirs(datasink)
        for dir in ['cmap', 'stat_map', 'zmap']:
            if not os.path.exists(os.path.join(datasink, dir)):
                os.mkdir(os.path.join(datasink, dir))

    # fit first level glm to estimate mean orientation
    mni_mask = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    fmri_glm = FirstLevelModel(t_r=3.0, slice_time_ref=0.5, hrf_model='spm',
                               drift_model=None, high_pass=1 / 100, mask_img=mni_mask,
                               smoothing_fwhm=None, verbose=1, n_jobs=1)
    fmri_glm = fmri_glm.fit(run_imgs, design_matrices=design_matrices)

    # define contrast
    contrasts = set_rsa_contrasts(design_matrices)

    # statistics inference
    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the two sessions

        stats_map = fmri_glm.compute_contrast(contrast_val, output_type='all')
        c_map = stats_map['effect_size']
        stat_map = stats_map['stat']
        z_map = stats_map['z_score']
        # write the resulting stat images to file
        c_image_path = join(datasink, 'cmap', '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)

        s_image_path = join(datasink, 'stat_map', '%s_smap.nii.gz' % contrast_id)
        stat_map.to_filename(s_image_path)

        z_image_path = join(datasink, 'zmap', '%s_zmap.nii.gz' % contrast_id)
        z_map.to_filename(z_image_path)


@memory.cache
def run_glm(subj):
    run_list = [1, 2, 3, 4, 5, 6]
    ifold = 6
    configs = {'TR': 3.0, 'task': 'game1', 'glm_type': 'grid_rsa',
               'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
               'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
               'func_name': r'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth8.nii',
               'events_name': r'sub-{}_task-game1_run-{}_events.tsv',
               'regressor_name': r'sub-{}_task-game1_run-{}_desc-confounds_timeseries.tsv'}

    dataroot = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn_test/{}/{}/Setall/{}fold'.format(configs['task'],
                                                                                         configs['glm_type'], ifold)
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    datasink = os.path.join(dataroot, 'sub-{}'.format(subj))
    if os.path.exists(datasink):
        print(f"sub-{subj} already have results.")
    else:
        print("-------{} start!--------".format(subj))
        functional_imgs, design_matrices = prepare_data(subj, run_list, ifold, configs, True)
        first_level_glm(datasink, functional_imgs, design_matrices)


if __name__ == "__main__":
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    subjects_chunk = list_to_chunk(subjects,1)
    for chunk in subjects_chunk:
        results_list = Parallel(n_jobs=80)(delayed(run_glm)(subj) for subj in chunk)
