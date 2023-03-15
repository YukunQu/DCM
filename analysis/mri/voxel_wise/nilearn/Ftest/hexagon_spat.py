# -*- coding: utf-8 -*-
"""

@author: QYK
"""
import os
import numpy as np
import pandas as pd
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk
from analysis.mri.voxel_wise.nilearn.firstLevel_analysis import prepare_data,get_reg_index,first_level_glm
from joblib import Parallel, delayed


def load_ev_spat(event_path):
    event = pd.read_csv(event_path, sep='\t')
    event_condition = event.query("trial_type in ['M1','M2','decision']")

    cos_mod = event.query("trial_type =='cos'")['modulation'].to_list()
    sin_mod = event.query("trial_type =='sin'")['modulation'].to_list()

    # generate parametric modulation for M2
    m2xcos = event.query("trial_type == 'M2'").copy()
    m2xcos.loc[:, 'modulation'] = cos_mod
    m2xcos['trial_type'] = 'M2xcos'

    m2xsin = event.query("trial_type == 'M2'").copy()
    m2xsin.loc[:, 'modulation'] = sin_mod
    m2xsin['trial_type'] = 'M2xsin'

    # generate parametric modulation for decision
    decisionxcos = event.query("trial_type == 'decision'").copy()
    decisionxcos.loc[:, 'modulation'] = cos_mod
    decisionxcos['trial_type'] = 'decisionxcos'

    decisionxsin = event.query("trial_type == 'decision'").copy()
    decisionxsin.loc[:, 'modulation'] = sin_mod
    decisionxsin['trial_type'] = 'decisionxsin'

    event_condition = event_condition.append([m2xcos,m2xsin,decisionxcos,decisionxsin])
    event_condition = event_condition[['onset', 'duration', 'trial_type', 'modulation']]
    return event_condition


def set_contrasts_spat(design_matrix):
    contrast_name = ['M1','M2','decision','M2xcos', 'M2xsin', 'decisionxcos', 'decisionxsin']
    # base contrast
    contrasts_set = {}
    for contrast_id in contrast_name:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector

    # advanced contrast
    contrasts_set['cos'] = contrasts_set['M2xcos'] + contrasts_set['decisionxcos']
    contrasts_set['sin'] = contrasts_set['M2xsin'] + contrasts_set['decisionxsin']

    contrasts_set['m2_hexagon'] = np.vstack([contrasts_set['M2xcos'], contrasts_set['M2xsin']])
    contrasts_set['decision_hexagon'] = np.vstack([contrasts_set['decisionxcos'], contrasts_set['decisionxsin']])
    contrasts_set['hexagon'] = np.vstack([contrasts_set['cos'], contrasts_set['sin']])
    return contrasts_set


def run_glm(task,subj):
    if task == 'game1':
        ifold = 6
        configs = {'TR': 3.0, 'task': 'game1', 'glm_type': 'hexagon_spat',
                   'run_list': [1, 2, 3, 4, 5, 6],
                   'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
                   'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
                   'func_name': 'func/sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii.gz',
                   'events_name': r'sub-{}_task-game1_run-{}_events.tsv',
                   'regressor_name': r'sub-{}_task-game1_run-{}_desc-confounds_timeseries_trimmed.tsv'}
    elif task == 'game2':
        ifold = 6
        configs = {'TR': 3.0, 'task': 'game2', 'glm_type': 'hexagon_spat',
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
        functional_imgs, design_matrices = prepare_data(subj, ifold, configs,load_ev_spat,True)
        first_level_glm(datasink, functional_imgs, design_matrices,set_contrasts_spat)


if __name__ == "__main__":
    task = 'game2'
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    subjects_chunk = list_to_chunk(subjects,2)
    for chunk in subjects_chunk:
        results_list = Parallel(n_jobs=12)(delayed(run_glm)(task,subj) for subj in chunk)
