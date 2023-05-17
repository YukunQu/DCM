# -*- coding: utf-8 -*-
"""
# distance effect
@author: QYK
"""
import os
import numpy as np
import pandas as pd
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk
from analysis.mri.voxel_wise.nilearn.firstLevel_analysis import prepare_data,get_reg_index,first_level_glm
from joblib import Parallel, delayed


def load_ev_distance(event_path):
    event = pd.read_csv(event_path, sep='\t')
    event_condition = event.query("trial_type in ['M1','M2_corr','M2_error', 'decision_corr','decision_error']")

    pmod_distance = event.query("trial_type=='distance'")
    distance_mod = pmod_distance['modulation'].to_list()

    # generate parametric modulation for M2
    m2xdistance = event.query("trial_type == 'M2_corr'").copy()
    m2xdistance.loc[:, 'modulation'] = distance_mod
    m2xdistance['trial_type'] = 'M2_corrxdistance'

    # # generate parametric modulation for decision
    decisionxdistance = event.query("trial_type == 'decision_corr'").copy()
    decisionxdistance.loc[:, 'modulation'] = distance_mod
    decisionxdistance['trial_type'] = 'decision_corrxdistance'

    event_condition = event_condition.append([m2xdistance,decisionxdistance])
    event_condition = event_condition[['onset', 'duration', 'trial_type', 'modulation']]
    return event_condition


def set_contrasts(design_matrix):
    contrast_name = ['M1','M2_corr','M2_error','decision_corr','decision_error','M2_corrxdistance','decision_corrxdistance']
    # base contrast
    contrasts_set = {}
    for contrast_id in contrast_name:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        if len(contrast_index) == 0:
            continue
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector

    # advanced contrast
    if 'decision_error' in contrasts_set.keys():
        contrasts_set['m2_correct_superiority'] = contrasts_set['M2_corr'] - contrasts_set['M2_error']
        contrasts_set['decision_correct_superiority'] = contrasts_set['decision_corr'] - contrasts_set['decision_error']
    return contrasts_set


def run_glm(task,subj,ifold):
    if task == 'game1':
        configs = {'TR': 3.0, 'task': 'game1', 'glm_type': 'distance_spct',
                   'run_list': [1, 2, 3, 4, 5, 6],
                   'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
                   'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
                   'func_name': 'func/sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii.gz',
                   'events_name': r'sub-{}_task-game1_run-{}_events.tsv',
                   'regressor_name': r'sub-{}_task-game1_run-{}_desc-confounds_timeseries_trimmed.tsv'}
    elif task == 'game2':
        configs = {'TR': 3.0, 'task': 'game2', 'glm_type': 'distance_center_spct',
                   'run_list': [1, 2],
                   'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
                   'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
                   'func_name': 'func/sub-{}_task-game2_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii.gz',
                   'events_name': r'sub-{}_task-game2_run-{}_events.tsv',
                   'regressor_name': r'sub-{}_task-game2_run-{}_desc-confounds_timeseries_trimmed.tsv'}
    else:
        raise Exception("The type of task is not supoort.")

    dataroot = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/{}/{}/Setall/{}fold'.format(configs['task'],
                                                                                       configs['glm_type'], ifold)
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    datasink = os.path.join(dataroot, 'sub-{}'.format(subj))
    if os.path.exists(datasink):
        print(f"sub-{subj} already have results.")
    else:
        print("-------{} start!--------".format(subj))
        functional_imgs, design_matrices = prepare_data(subj,ifold,configs,load_ev_distance,concat_runs=True,despiking=True)
        first_level_glm(datasink, functional_imgs, design_matrices, set_contrasts)


if __name__ == "__main__":
    task = 'game1'
    ifold = 6
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    subjects_chunk = list_to_chunk(subjects,20)
    for chunk in subjects_chunk:
        results_list = Parallel(n_jobs=20)(delayed(run_glm)(task,subj,ifold) for subj in chunk)
