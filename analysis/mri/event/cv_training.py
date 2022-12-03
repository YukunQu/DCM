#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:28:44 2022

@author: dell
"""

import os
import random
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV


class Game1_cv(Game1EV):
    def __init__(self, behDataPath):
        self.starttime = None
        Game1EV.__init__(self,behDataPath)

    def _split_corr_trials(self, trial_label):
        i = 0
        odd_trial = []
        even_trial = []
        for label in trial_label:
            if label:
                i += 1
                if i % 2 == 0:
                    odd_trial.append(None)
                    even_trial.append(label)
                else:
                    odd_trial.append(label)
                    even_trial.append(None)
            else:
                odd_trial.append(label)
                even_trial.append(label)
        return odd_trial, even_trial

    def _split_whole_trials(self, trial_label):
        i = 0
        odd_trial = []
        even_trial = []
        for label in trial_label:
            i += 1
            if i % 2 == 0:
                odd_trial.append(None)
                even_trial.append(label)
            else:
                odd_trial.append(label)
                even_trial.append(None)
        return odd_trial, even_trial

    def split_trials(self, trial_label, trial_type, split_target='corr_trials'):
        if split_target == 'corr_trials':
            odd_trial, even_trial = self._split_corr_trials(trial_label)
        elif split_target == 'whole_trials':
            odd_trial, even_trial = self._split_whole_trials(trial_label)
        else:
            raise Exception("The split target is not supported.")

        if trial_type == 'odd':
            return odd_trial
        elif trial_type == 'even':
            return even_trial
        elif trial_type == 'both':
            return odd_trial, even_trial

    def genpm_train(self, m2ev_corr, ifold):
        angle = m2ev_corr['angle']
        pmod_sin = m2ev_corr.copy()
        pmod_cos = m2ev_corr.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold * angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold * angle))
        return pmod_sin, pmod_cos

    def genpm_test(self, m2ev_corr, ifold, phi):
        angle = m2ev_corr['angle']
        pmod_alignPhi = m2ev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game1_cv_train(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        odd_trial, even_trial = self.split_trials(trial_label, 'both')

        # even event
        m2ev_corr, m2ev_error = self.genM2ev(even_trial)
        deev_corr, deev_error = self.genDeev(even_trial)
        pmod_sin, pmod_cos = self.genpm_train(m2ev_corr, ifold)

        even_event = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_sin, pmod_cos], axis=0)

        # odd event
        m2ev_corr, m2ev_error = self.genM2ev(odd_trial)
        deev_corr, deev_error = self.genDeev(odd_trial)
        pmod_sin, pmod_cos = self.genpm_train(m2ev_corr, ifold)

        odd_event = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                               pmod_sin, pmod_cos], axis=0)
        return even_event, odd_event

    def game1_cv_test(self, ifold, phi, trial_type):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        trial_set = self.split_trials(trial_label, trial_type)

        # generate event of trials
        m2ev_corr, m2ev_error = self.genM2ev(trial_set)
        deev_corr, deev_error = self.genDeev(trial_set)
        pmod_alignPhi = self.genpm_test(m2ev_corr, ifold, phi)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error, pmod_alignPhi], axis=0)
        return event_data


def gen_event_game1_cv_train():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query("(game1_fmri==0.5)or(game1_fmri==0.7)")
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]
    subject_list = ['172']

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/separate_hexagon_correct_trials_train/sub-{}/{}fold'
    event_file = 'sub-{}_task-game1_run-{}_set-{}_events.tsv'

    # set folds and runs for cross validation
    ifolds = range(4, 9)
    runs = range(1, 7)

    for sub in subject_list:
        print(sub, "started.")
        for ifold in ifolds:
            for run_id in runs:
                # generate event
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv(behDataPath)
                even_event, odd_event = game1_cv.game1_cv_train(ifold)
                # save
                out_dir = save_dir.format(sub, ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path1 = join(out_dir, event_file.format(sub, run_id, 'even'))
                tsv_save_path2 = join(out_dir, event_file.format(sub, run_id, 'odd'))
                even_event.to_csv(tsv_save_path1, sep="\t", index=False)
                odd_event.to_csv(tsv_save_path2, sep="\t", index=False)


def gen_event_game1_cv_test():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query("(game1_fmri==0.5)or(game1_fmri==0.7)")
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/' \
               r'alignPhi_separate_correct_trials/sub-{}/{}fold'  # look out
    event_file = 'sub-{}_task-game1_run-{}_set-{}_events.tsv'

    # set Phi estimated from specific ROI
    odd_phi_file = r'/mnt/workdir/DCM/result/CV/Phi/estPhi_ROI-mPFC_On-m2_Set2.csv'  # look out
    odd_phi_data = pd.read_csv(odd_phi_file)

    even_phi_file = r'/mnt/workdir/DCM/result/CV/Phi/estPhi_ROI-mPFC_On-m2_Set1.csv'  # look out
    even_phi_data = pd.read_csv(even_phi_file)

    # set folds and runs for cross validation
    ifolds = range(6, 7)
    runs = range(1, 7)

    for sub in subject_list:
        print(sub, "started.")
        for ifold in ifolds:
            odd_phi = odd_phi_data.query(f'(sub_id=="sub-{sub}")and(ifold=="{ifold}fold")')['Phi'].values[0]
            even_phi = even_phi_data.query(f'(sub_id=="sub-{sub}")and(ifold=="{ifold}fold")')['Phi'].values[0]
            for run_id in runs:
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv(behDataPath)
                odd_event = game1_cv.game1_cv_test(ifold, even_phi, 'odd')
                even_event = game1_cv.game1_cv_test(ifold, odd_phi, 'even')
                # save
                out_dir = save_dir.format(sub, ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path = join(out_dir, event_file.format(sub, run_id, 'odd'))
                odd_event.to_csv(tsv_save_path, sep="\t", index=False)
                tsv_save_path = join(out_dir, event_file.format(sub, run_id, 'even'))
                even_event.to_csv(tsv_save_path, sep='\t', index=False)


def Test_gen_event_game1_cv_train():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]
    subject_list = random.sample(subject_list, 3)
    pass


gen_event_game1_cv_test()
