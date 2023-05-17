# -*- coding: utf-8 -*-
"""

@author: QYK
"""
import os
import numpy as np
import pandas as pd
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk
from analysis.mri.voxel_wise.nilearn.firstLevel_analysis import load_ev, prepare_data, get_reg_index, first_level_glm
from joblib import Parallel, delayed


def set_contrasts(design_matrix):
    # set contrast
    contrast_name = ['M1', 'M2_error', 'decision_error']
    for onset in ['m2', 'decision']:
        for even_odd in ['even', 'odd']:
            for bin in range(1, 13, 1):
                if bin in range(1, 13, 2):
                    contrast_name.append(onset + '_align_' + str(bin) + "_" + even_odd)
                elif bin in range(2, 13, 2):
                    contrast_name.append(onset + '_misalign_' + str(bin) + "_" + even_odd)

    # base contrast
    contrasts_set = {}
    for contrast_id in contrast_name:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        if len(contrast_index) == 0:
            print(contrast_id, ' have no regressor!')
            continue

        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector

    # advanced contrast
    # align and misalign and alignPhi contrast
    for onset in ['m2', 'decision']:
        for odevity in ['even', 'odd']:
            align = np.zeros(design_matrix.shape[1])
            missalign = np.zeros(design_matrix.shape[1])
            for cid, cvt in contrasts_set.items():
                if (cid.startswith(f'{onset}_align')) and (odevity in cid):
                    align += cvt
                elif (cid.startswith(f'{onset}_misalign')) and (odevity in cid):
                    missalign += cvt
                else:
                    continue
            contrasts_set[f'{onset}_align_{odevity}'] = align
            contrasts_set[f'{onset}_misalign_{odevity}'] = missalign
            contrasts_set[f'{onset}_alignPhi_{odevity}'] = align - missalign
        contrasts_set[f'{onset}_align'] = contrasts_set[f'{onset}_align_odd'] + contrasts_set[f'{onset}_align_even']
        contrasts_set[f'{onset}_misalign'] = contrasts_set[f'{onset}_misalign_odd'] + contrasts_set[f'{onset}_misalign_even']
        contrasts_set[f'{onset}_alignPhi'] = contrasts_set[f'{onset}_align'] - contrasts_set[f'{onset}_misalign']
    contrasts_set['alignPhi'] = contrasts_set['m2_alignPhi'] + contrasts_set['decision_alignPhi']
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
        functional_imgs, design_matrices = prepare_data(subj, ifold, configs, load_ev, True, False)
        first_level_glm(datasink, functional_imgs, design_matrices, set_contrasts)


if __name__ == "__main__":
    # specify configure parameters
    configs = {'TR': 3.0,
               'task': 'game1',
               'glm_type': 'cv_test_12bin_spct',
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

    subjects_chunk = list_to_chunk(subjects, 50)
    for ifold in range(6, 7):
        # creat dataroot
        dataroot = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/{}/{}/Setall/{}fold'.format(configs['task'],
                                                                                           configs['glm_type'], ifold)
        if not os.path.exists(dataroot):
            os.makedirs(dataroot)

        configs['ifold'] = ifold
        configs['dataroot'] = dataroot
        for chunk in subjects_chunk:
            results_list = Parallel(n_jobs=50)(delayed(run_glm)(subj, configs) for subj in chunk)
