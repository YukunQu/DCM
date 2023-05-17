# -*- coding: utf-8 -*-
"""

@author: QYK
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk
from analysis.mri.voxel_wise.nilearn.firstLevel_analysis import prepare_data, get_reg_index, first_level_glm


def load_ev(event_path):
    # load behavioral file and generate event without distance
    event = pd.read_csv(event_path, sep='\t')
    regressors_name = ['M1', 'M2_corr', 'M2_error', 'decision_corr', 'decision_error',
                       'cos_even', 'cos_odd', 'sin_even', 'sin_odd']
    event = event[event['trial_type'].isin(regressors_name)]
    event_condition = event[['onset', 'duration', 'trial_type', 'modulation']]
    return event_condition


def set_contrasts(design_matrix):
    # set contrast contain hexagonal effect and distance effect
    contrast_name = ['M1', 'M2_corr', 'M2_error', 'decision_corr', 'decision_error',
                     'cos_even', 'cos_odd', 'sin_even', 'sin_odd']
    # base contrast
    contrasts_set = {}
    for contrast_id in contrast_name:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector

    # advanced contrast
    # odd trials' hexagonal modulation
    contrasts_set['odd_hexagon'] = np.vstack([contrasts_set['cos_odd'],
                                              contrasts_set['sin_odd']])
    # even trials' hexagonal modulation
    contrasts_set['even_hexagon'] = np.vstack([contrasts_set['cos_even'],
                                               contrasts_set['sin_even']])

    # all trials' hexagonal modulation
    contrasts_set['cos'] = contrasts_set['cos_odd'] + contrasts_set['cos_even']
    contrasts_set['sin'] = contrasts_set['sin_odd'] + contrasts_set['sin_even']
    contrasts_set['hexagon'] = np.vstack([contrasts_set['cos'],
                                          contrasts_set['sin']])

    # correct contrast to error
    if 'decision_error' in contrasts_set.keys():
        contrasts_set['m2_correct_superiority'] = contrasts_set['M2_corr'] - contrasts_set['M2_error']
        contrasts_set['decision_correct_superiority'] = contrasts_set['decision_corr'] - contrasts_set['decision_error']
    return contrasts_set


def run_glm(subj, configs):
    # read parameters
    dataroot = configs['dataroot']
    ifold = configs['ifold']
    # skill the subject who already have results
    datasink = os.path.join(dataroot, 'sub-{}'.format(subj))
    if os.path.exists(datasink):
        print(f"sub-{subj} already have results.")
    else:
        print("-------{} start!--------".format(subj))
        functional_imgs, design_matrices = prepare_data(subj, ifold, configs, load_ev, concat_runs=True, despiking=False)
        first_level_glm(datasink, functional_imgs, design_matrices, set_contrasts)


if __name__ == "__main__":
    # specify configure parameters
    configs = {'TR': 3.0,
               'task': 'game1',
               'glm_type': 'cv_train_hexagon_spct',
               'run_list': [1, 2, 3, 4, 5, 6],
               'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
               'events_name': r'sub-{}_task-game1_run-{}_events.tsv',
               'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
               'func_name': 'func/sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii.gz',
               'regressor_name': r'sub-{}_task-game1_run-{}_desc-confounds_timeseries_trimmed.tsv'}

    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{configs["task"]}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    subjects_chunk = list_to_chunk(subjects, 60)
    for ifold in [6]:
        # creat dataroot
        dataroot = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/{}/{}/Setall/{}fold'.format(configs['task'],
                                                                                           configs['glm_type'], ifold)
        if not os.path.exists(dataroot):
            os.makedirs(dataroot)

        configs['ifold'] = ifold
        configs['dataroot'] = dataroot
        for chunk in subjects_chunk:
            results_list = Parallel(n_jobs=60)(delayed(run_glm)(subj, configs) for subj in chunk)