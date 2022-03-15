#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:37:30 2022

@author: dell
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from os.path import join as pjoin

from nilearn.image import load_img
from nilearn.maskers import NiftiMasker


def extract_activity(func, roi, confounds_file, timestamp=True, event=None):
    # extract activity from ROI based on event file
    motion_columns = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                      'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                      'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
                      'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
                      'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                      'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
    confounds = confounds_file[motion_columns].copy()
    confounds.fillna(0, inplace=True)
    brain_masker = NiftiMasker(mask_img=roi, smoothing_fwhm=8, detrend=True, standardize=False, high_pass=1/128, t_r=3)
    activity_time_series = np.nanmean(brain_masker.fit_transform(func, confounds=confounds), axis=1)

    bold_timestamp = []
    if timestamp:
        for i, activity in enumerate(activity_time_series):
            bold_timestamp.append(1.46 + i * 3)  # tr = 3

    brain_time_series = pd.DataFrame({'bold': activity_time_series,
                                      'timestamp': bold_timestamp})

    if event is not None:
        m2s_activities = pd.DataFrame(columns=['bold', 'timestamp', 'angle'])
        for row in event.itertuples():
            onset = row.onset
            duration = row.duration
            m2_activity = brain_time_series.query('{}<timestamp<={}'.format(onset, onset + duration)).copy()
            m2_activity.loc[:, 'angle'] = row.angle
            m2s_activities = m2s_activities.append(m2_activity, ignore_index=True)
        m2s_activities[['bold']] = m2s_activities[['bold']].apply(zscore)
        return m2s_activities
    else:
        brain_time_series[['bold']] = brain_time_series[['bold']].apply(zscore)
        return brain_time_series


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query('usable==1')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('_')[-1] for p in pid][20:]
    subjects = ['065']
    runs = [1, 2, 3, 4, 5, 6]

    data_root = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume'
    templates = {'func': pjoin(data_root, 'sub-{sub_id}/func/'
                                          'sub-{sub_id}_task-game1_run-'
                                          '{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),

                 'confounds': pjoin(data_root, 'sub-{sub_id}/func/'
                                               'sub-{sub_id}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv'),

                 'event': pjoin(r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{sub_id}/hexonM2Long/6fold',
                                'sub-{sub_id}_task-game1_run-{run_id}_events.tsv'),

                 'EC': r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/defROI/adult/EC_func_roi.nii',

                 'vmPFC': r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/defROI/adult/vmpfc_func_roi.nii'}

    for sub_id in subjects:
        print(sub_id, " start.")
        for roi_name in ['EC', 'vmPFC']:
            sub_activity = pd.DataFrame(columns=['sub_id', 'run_id', 'bold', 'timestamp', 'angle'])

            for run_id in runs:
                func = templates['func'].format(sub_id=sub_id, run_id=run_id)
                confounds_file = pd.read_csv(templates['confounds'].format(sub_id=sub_id, run_id=run_id), sep='\t')
                event = pd.read_csv(templates['event'].format(sub_id=sub_id, run_id=run_id),
                                    sep='\t').query('trial_type=="M2_corr"')
                roi = load_img(templates[roi_name])

                sub_roi_activities = extract_activity(func, roi, confounds_file, timestamp=True, event=event)
                sub_roi_activities['sub_id'] = 'sub-{}'.format(sub_id.zfill(3))
                sub_roi_activities['run_id'] = run_id
                sub_activity = sub_activity.append(sub_roi_activities, ignore_index=False)

            sub_dir = pjoin(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/brain_activity',
                            'sub-{sub_id}').format(sub_id=sub_id)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            savePath = pjoin(sub_dir, 'sub-{sub_id}_{roi}_M2_activity.csv'.format(sub_id=sub_id, roi=roi_name))
            sub_activity.to_csv(savePath, index=False)