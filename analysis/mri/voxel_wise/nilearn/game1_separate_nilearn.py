# -*- coding: utf-8 -*-
"""

@author: QYK
"""
import os
import numpy as np
import pandas as pd
from os.path import join
from nilearn.image import math_img, load_img,resample_to_img
from nilearn.masking import apply_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix


def load_ev_separate(event_path):
    event = pd.read_csv(event_path,sep='\t')
    event_condition = event.query("trial_type in ['M1','M2_corr','M2_error','decision_corr','decision_error']")

    pmod_cos = event.query("trial_type =='cos'")
    pmod_sin = event.query("trial_type =='sin'")

    # generate parametric modulation for M2
    m2_corrxcos = pmod_cos.copy()
    m2_corrxcos['trial_type'] = 'M2_corrxcos'

    m2_corrxsin = pmod_sin.copy()
    m2_corrxsin['trial_type'] = 'M2_corrxsin'

    # generate parametric modulation for decision
    cos_mod = pmod_cos['modulation'].to_list()
    sin_mod = pmod_sin['modulation'].to_list()

    decision_corrxcos = event.query("trial_type == 'decision_corr'")
    decision_corrxsin = decision_corrxcos.copy()

    decision_corrxcos = decision_corrxcos.replace('decision_corr','decision_corrxcos')
    decision_corrxsin = decision_corrxsin.replace('decision_corr','decision_corrxsin')

    decision_corrxcos.loc[:,'modulation'] = cos_mod
    decision_corrxsin.loc[:,'modulation'] = sin_mod

    event_condition = event_condition.append([m2_corrxcos,m2_corrxsin,decision_corrxcos,decision_corrxsin])
    event_condition = event_condition[['onset', 'duration', 'trial_type', 'modulation']]
    return event_condition


def prepare_data(subj,run_list,ifold,configs):
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
    for i,run_id in enumerate(run_list):
        # load image
        func_path = join(func_dir,f'sub-{subj}','func',func_name.format(subj,run_id))
        func_img = load_img(func_path)
        functional_imgs.append(func_img)

        # load event
        event_path = join(event_dir,task,glm_type,f'sub-{subj}',ifold,events_name.format(subj,run_id))
        event = load_ev_separate(event_path)

        # load motion
        add_reg_names =  ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                          'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                          'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
                          'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
                          'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                          'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
        confound_file = os.path.join(func_dir,f'sub-{subj}', 'func',regressor_file.format(subj,run_id))
        confound_factors = pd.read_csv(confound_file,sep="\t")
        motion = confound_factors[add_reg_names]
        motion = motion.fillna(0.0)

        # creat design matrix
        n_scans = func_img.shape[-1]
        frame_times = np.arange(n_scans) * tr
        high_pass_fre = 1/100
        design_matrix = make_first_level_design_matrix(
            frame_times,
            event,
            hrf_model='spm',
            drift_model=None,
            high_pass=high_pass_fre,add_regs=motion,add_reg_names=add_reg_names)
        design_matrices.append(design_matrix)

    return functional_imgs, design_matrices


def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def set_contrasts(design_matrices):
    regressors_num = set([len(dm) for dm in design_matrices])
    if regressors_num == 1:
        contrast_matrix = np.eye(design_matrices[0].shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrices[0].columns)])

        contrasts = {'m2_cos':basic_contrasts['M2_corrxcos'],
                     'm2_sin':basic_contrasts['M2_corrxsin'],
                     'decision_cos':basic_contrasts['decision_corrxcos'],
                     'decision_sin':basic_contrasts['decision_corrxsin'],
                     'm2':basic_contrasts['M2_corr'],
                     'decision':basic_contrasts['decision_corr']}

        contrasts['cos'] = contrasts['m2_cos'] + contrasts['decision_cos']
        contrasts['sin'] = contrasts['m2_sin'] + contrasts['decision_sin']

        contrasts['m2_hexagon'] = np.vstack([contrasts['m2_cos'],contrasts['m2_sin']])
        contrasts['decision_hexagon'] = np.vstack([contrasts['decision_cos'],contrasts['decision_sin']])
        contrasts['hexagon'] = np.vstack([contrasts['cos'],contrasts['sin']])
    else:
        print("The regressors_num is not equal between runs.")
        contrasts = {'m2_cos':[],'m2_sin':[],'decision_cos':[],'decision_sin':[],'m2':[],'decision':[],
                     'cos':[],'sin':[],'m2_hexagon':[],'decision_hexagon':[],'hexagon':[]}

        for design_matrix in design_matrices:
            contrast_matrix = np.eye(design_matrix.shape[1])
            basic_contrasts = dict([(column, contrast_matrix[i])
                                    for i, column in enumerate(design_matrix.columns)])

            m2_cos = basic_contrasts['M2_corrxcos']
            m2_sin = basic_contrasts['M2_corrxsin']
            decision_cos = basic_contrasts['decision_corrxcos']
            decision_sin = basic_contrasts['decision_corrxsin']
            m2 = basic_contrasts['M2_corr']
            decision = basic_contrasts['decision_corr']

            cos = m2_cos + decision_cos
            sin = m2_sin + decision_sin

            m2_hexagon = np.vstack([m2_cos,m2_sin])
            decision_hexagon = np.vstack([decision_cos,decision_sin])
            hexagon = np.vstack([cos,sin])

            for contrast_id in ['m2_cos','m2_sin','decision_cos','decision_sin','m2','decision',
                                'cos','sin','m2_hexagon','decision_hexagon','hexagon']:
                contrasts[contrast_id].append(eval(contrast_id))
    return contrasts


def first_level_glm(datasink,run_imgs,design_matrices):
    # fit first level glm to estimate mean orientation
    mni_mask = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    fmri_glm = FirstLevelModel(t_r=3.0,slice_time_ref=0.5,hrf_model='spm',
                               drift_model=None,high_pass=1/128,mask_img=mni_mask,
                               smoothing_fwhm=8,verbose=1,n_jobs=10)
    fmri_glm = fmri_glm.fit(run_imgs, design_matrices=design_matrices)

    # define contrast
    contrast_matrix = np.eye(design_matrices[0].shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrices[0].columns)])

    contrasts = set_contrasts(design_matrices)

    # statistics inference
    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the two sessions

        stats_map = fmri_glm.compute_contrast(contrast_val, output_type='all')
        c_map = stats_map['effect_size']
        t_map = stats_map['stat']
        z_map = stats_map['z_score']

        # write the resulting stat images to file
        if not os.path.exists(datasink):
            os.makedirs(datasink)
        c_image_path = join(datasink, 'sub-%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)

        t_image_path = join(datasink, 'sub-%s_tmap.nii.gz' % contrast_id)
        t_map.to_filename(t_image_path)

        z_image_path = join(datasink, 'sub-%s_zmap.nii.gz' % contrast_id)
        z_map.to_filename(z_image_path)


if __name__ == "__main__":
    run_list = [1,2,3,4,5,6]
    ifold = 6
    configs = {'TR':3.0, 'task':'game1', 'glm_type':'separate_hexagon',
               'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep',
               'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
               'func_name': r'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
               'events_name':r'sub-{}_task-game1_run-{}_events.tsv',
               'regressor_name':r'sub-{}_task-game1_run-{}_desc-confounds_timeseries.tsv'}

    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]
    aleary_subs = os.listdir('/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/separate_hexagon/Setall/6fold')
    aleary_subs = [a.split('-')[-1] for a in aleary_subs]

    for subj in subjects:
        if subj in aleary_subs:
            continue
        print("-------{} start!--------".format(subj))
        datasink = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/{}/{}/Setall/{}fold/sub-{}'.format(configs['task'],
                                                                                                  configs['glm_type'],
                                                                                                  ifold,subj)
        functional_imgs, design_matrices = prepare_data(subj,run_list,ifold,configs)
        first_level_glm(datasink,functional_imgs,design_matrices)
