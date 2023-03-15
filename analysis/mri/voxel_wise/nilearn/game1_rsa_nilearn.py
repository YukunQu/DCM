# -*- coding: utf-8 -*-
"""
# distance effect
@author: QYK
"""
import os
import numpy as np
import pandas as pd
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk
from analysis.mri.voxel_wise.nilearn.firstLevel_analysis import load_ev,prepare_data,get_reg_index,first_level_glm
from joblib import Parallel, delayed


def set_rsa_contrasts(design_matrix):
    regressor = design_matrix.columns
    angle_regs = list(set([reg.split("_")[-1] for reg in regressor if 'angle' in reg]))
    angle_regs.sort()
    contrast_name = ['M1','M2_error','decision'] + angle_regs

    # base contrast
    contrasts_set = {}
    for contrast_id in contrast_name:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector
    return contrasts_set


def run_glm(subj,configs,ifold):
    dataroot = r'{}/{}/{}/Setall/{}fold'.format(configs['out_rootdir'],configs['task'],configs['glm_type'], ifold)
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    datasink = os.path.join(dataroot, 'sub-{}'.format(subj))
    if os.path.exists(datasink):
        print(f"sub-{subj} already have results.")
    else:
        print("-------{} start!--------".format(subj))
        functional_imgs, design_matrices = prepare_data(subj,ifold,configs,load_ev,True)
        first_level_glm(datasink, functional_imgs, design_matrices, set_rsa_contrasts)


if __name__ == "__main__":
    task = 'game1'
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    if task == 'game1':
        configs = {'TR': 3.0, 'task': 'game1', 'glm_type': 'grid_rsa_corr_trials',
                   'run_list': [1, 2, 3, 4, 5, 6],
                   'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
                   'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
                   'func_name': 'func/sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii.gz',
                   'events_name': r'sub-{}_task-game1_run-{}_events.tsv',
                   'regressor_name': r'sub-{}_task-game1_run-{}_desc-confounds_timeseries_trimmed.tsv',
                   'out_rootdir':'/mnt/workdir/DCM/BIDS/derivatives/Nilearn'}
    elif task == 'game2':
        configs = {'TR': 3.0, 'task': 'game2', 'glm_type': 'grid_rsa_corr_trials',
                   'run_list': [1, 2],
                   'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
                   'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
                   'func_name': 'func/sub-{}_task-game2_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii.gz',
                   'events_name': r'sub-{}_task-game2_run-{}_events.tsv',
                   'regressor_name': r'sub-{}_task-game2_run-{}_desc-confounds_timeseries_trimmed.tsv',
                   'out_rootdir':'/mnt/workdir/DCM/BIDS/derivatives/Nilearn'}
    else:
        raise Exception("The type of task is not supoort.")

    ifold = 6
    subjects_chunk = list_to_chunk(subjects,70)
    for chunk in subjects_chunk:
        results_list = Parallel(n_jobs=70)(delayed(run_glm)(subj,configs,ifold) for subj in chunk)