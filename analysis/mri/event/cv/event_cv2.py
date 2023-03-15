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
from analysis.mri.event.distance.event_distance import Game1EV_hexagon_distance_spct


class Game1_cv(Game1EV_hexagon_distance_spct):
    def __init__(self, behDataPath):
        Game1EV_hexagon_distance_spct.__init__(self, behDataPath)

    @staticmethod
    def _split_corr_trials(trial_corr):
        i = 0
        trial_label = []
        for tc in trial_corr:
            if tc:
                i += 1
                if i % 2 == 0:
                    trial_label.append('even')
                else:
                    trial_label.append('odd')
            else:
                trial_label.append('error')
        return trial_label

    @staticmethod
    def _split_whole_trials(trial_corr):
        trial_label = []
        for i in range(1, len(trial_corr) + 1):
            if i % 2 == 0:
                trial_label.append('even')
            else:
                trial_label.append('odd')
        return trial_label

    def split_trials(self, trial_corr, split_target='corr_trials'):
        if split_target == 'corr_trials':
            trial_label = self._split_corr_trials(trial_corr)
        elif split_target == 'whole_trials':
            trial_label = self._split_whole_trials(trial_corr)
        else:
            raise Exception("The split target is not supported.")
        return trial_label

    def genM2ev(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        m2ev_corr_odd = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        m2ev_corr_even = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        m2ev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(m2ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        for i, label in enumerate(trial_label):
            if label == 'odd':
                m2ev_corr_odd = m2ev_corr_odd.append(m2ev.iloc[i])
            elif label == 'even':
                m2ev_corr_even = m2ev_corr_even.append(m2ev.iloc[i])
            elif label == 'error':
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be True,False or None.")
        m2ev_corr_odd['trial_type'] = 'M2_corr_odd'
        m2ev_corr_even['trial_type'] = 'M2_corr_even'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr_odd = m2ev_corr_odd.sort_values('onset', ignore_index=True)
        m2ev_corr_even = m2ev_corr_even.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr_odd, m2ev_corr_even, m2ev_error

    def genM2ev_split_whole_trials(self, trial_run_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        m2ev_odd = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        m2ev_even = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(m2ev) == len(trial_run_label), "The number of trial label didn't not same as the number of event-M2."

        for i,label in enumerate(trial_run_label):
            if label == 'odd':
                m2ev_odd = m2ev_odd.append(m2ev.iloc[i])
            elif label == 'even':
                m2ev_even = m2ev_even.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be odd or even.")
        m2ev_odd['trial_type'] = 'M2_odd'
        m2ev_even['trial_type'] = 'M2_even'

        m2ev_odd = m2ev_odd.sort_values('onset', ignore_index=True)
        m2ev_even = m2ev_even.sort_values('onset', ignore_index=True)
        return m2ev_odd, m2ev_even

    def genDeev(self, trial_label):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        deev_corr_odd = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        deev_corr_even = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        deev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(deev) == len(
            trial_label), "The number of trial label didn't not  same as the number of event-decision."

        for i, trial_label in enumerate(trial_label):
            if trial_label == 'odd':
                deev_corr_odd = deev_corr_odd.append(deev.iloc[i])
            elif trial_label == 'even':
                deev_corr_even = deev_corr_even.append(deev.iloc[i])
            elif trial_label == 'error':
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr_odd['trial_type'] = 'decision_corr_odd'
        deev_corr_even['trial_type'] = 'decision_corr_even'
        deev_error['trial_type'] = 'decision_error'

        deev_corr_odd = deev_corr_odd.sort_values('onset', ignore_index=True)
        deev_corr_even = deev_corr_even.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_corr_odd, deev_corr_even, deev_error

    def genDeev_split_whole_trials(self, trial_corr, trial_label):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        deev_odd = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        deev_even = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        deev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(deev) == len(
            trial_label), "The number of trial label didn't not  same as the number of event-decision."
        assert len(deev) == len(
            trial_corr), "The number of trial label didn't not  same as the number of event-decision."

        for i, corr, label in enumerate(trial_corr, trial_label):
            if corr:
                if label == 'odd':
                    deev_odd = deev_odd.append(deev.iloc[i])
                elif label == 'even':
                    deev_even = deev_even.append(deev.iloc[i])
            elif not corr:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_odd['trial_type'] = 'decision_odd_corr'
        deev_even['trial_type'] = 'decision_even_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_odd = deev_odd.sort_values('onset', ignore_index=True)
        deev_even = deev_even.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_odd, deev_even, deev_error

    def genM2ev_without_pmod(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m2ev = m2ev.sort_values('onset', ignore_index=True)
        return m2ev

    def genDeev_without_pmod(self):
        # generate the event of decision without discriminate (corr or error)
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        deev = deev.sort_values('onset', ignore_index=True)
        return deev

    def genpm_train(self, ev, ifold, trial_type):
        angle = ev['angle']
        pmod_sin = ev.copy()
        pmod_cos = ev.copy()
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold * angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold * angle))
        if trial_type == 'odd':
            pmod_sin['trial_type'] = 'sin_odd'
            pmod_cos['trial_type'] = 'cos_odd'
        elif trial_type == 'even':
            pmod_sin['trial_type'] = 'sin_even'
            pmod_cos['trial_type'] = 'cos_even'
        else:
            raise Exception("The trial type only support odd and even.")
        return pmod_sin, pmod_cos

    def genpm_test(self, ev, ifold, phi, trial_type):
        angle = ev['angle']
        pmod_alignPhi = ev.copy()
        if trial_type == 'odd':
            pmod_alignPhi['trial_type'] = 'alignPhi_odd'
        elif trial_type == 'even':
            pmod_alignPhi['trial_type'] = 'alignPhi_even'
        else:
            raise Exception("The variable 'trial type' only support odd and 'even'")
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game1_cv_train1(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2
        trial_corr, accuracy = self.label_trial_corr()
        trial_label = self.split_trials(trial_corr)
        m2ev_corr_odd, m2ev_corr_even, m2ev_error = self.genM2ev(trial_label)
        # pmod
        pmod_sin_odd, pmod_cos_odd = self.genpm_train(m2ev_corr_odd, ifold, 'odd')
        pmod_sin_even, pmod_cos_even = self.genpm_train(m2ev_corr_even, ifold, 'even')
        # decision
        decision = self.genDeev_whole_trials()
        event = pd.concat([m1ev, m2ev_corr_odd, m2ev_corr_even, m2ev_error, decision,
                           pmod_sin_odd, pmod_cos_odd, pmod_sin_even, pmod_cos_even], axis=0)
        return event

    def game1_cv_train2(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2
        m2ev = self.genM2ev_without_pmod()
        # decision
        trial_corr, accuracy = self.label_trial_corr()
        trial_label = self.split_trials(trial_corr)
        deev_corr_odd, deev_corr_even, deev_error = self.genDeev(trial_label)
        # pmod
        pmod_sin_odd, pmod_cos_odd = self.genpm_train(deev_corr_odd, ifold, 'odd')
        pmod_sin_even, pmod_cos_even = self.genpm_train(deev_corr_even, ifold, 'even')

        event = pd.concat([m1ev, m2ev, deev_corr_odd, deev_corr_even, deev_error,
                           pmod_sin_odd, pmod_cos_odd, pmod_sin_even, pmod_cos_even], axis=0)
        return event


    def genDeev_corr(self, trial_corr):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        deev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        deev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(deev) == len(
            trial_corr), "The number of trial label didn't not  same as the number of event-decision."

        for i, label in enumerate(trial_corr):
            if label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif label == False:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_corr, deev_error


    def game1_cv_train3(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2
        trial_corr, accuracy = self.label_trial_corr()
        trial_label = self.split_trials(trial_corr, 'whole_trials')
        m2ev_odd, m2ev_even = self.genM2ev_split_whole_trials(trial_label)
        # pmod
        pmod_sin_odd, pmod_cos_odd = self.genpm_train(m2ev_odd, ifold, 'odd')
        pmod_sin_even, pmod_cos_even = self.genpm_train(m2ev_even, ifold, 'even')
        # decision
        decision_corr, decision_error = self.genDeev_corr(trial_corr)
        # pressButton
        response = self.response()
        event = pd.concat([m1ev, m2ev_odd, m2ev_even, decision_corr, decision_error, response,
                           pmod_sin_odd, pmod_cos_odd, pmod_sin_even, pmod_cos_even], axis=0)
        return event

    def game1_cv_train_M2(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        # m2
        trial_corr, accuracy = self.label_trial_corr()
        trial_run_label = self.split_trials(trial_corr, 'whole_trials')
        m2ev_odd, m2ev_even = self.genM2ev_split_whole_trials(trial_run_label)
        # pmod
        pmod_sin_odd, pmod_cos_odd = self.genpm_train(m2ev_odd, ifold, 'odd')
        pmod_sin_even, pmod_cos_even = self.genpm_train(m2ev_even, ifold, 'even')
        # decision
        deev = self.genDeev_whole_trials()
        # respsone
        response = self.response()
        event = pd.concat([m1ev, m2ev_odd, m2ev_even, deev, response,
                           pmod_sin_odd, pmod_cos_odd, pmod_sin_even, pmod_cos_even], axis=0)
        return event

    def game1_cv_test_m2(self, ifold, odd_phi, even_phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2 &
        trial_corr, accuracy = self.label_trial_corr()
        trial_run_label = self.split_trials(trial_corr, 'whole_trials')
        m2ev_odd, m2ev_even = self.genM2ev_split_whole_trials(trial_run_label)
        # decision
        deev = self.genDeev_whole_trials()
        # respsone
        response = self.response()
        # pmod
        pmod_alignPhi_odd = self.genpm_test(m2ev_odd, ifold, even_phi, 'odd')
        pmod_alignPhi_even = self.genpm_test(m2ev_even, ifold, odd_phi, 'even')

        event_data = pd.concat([m1ev, m2ev_odd, m2ev_even, deev, response,
                                pmod_alignPhi_odd, pmod_alignPhi_even], axis=0)
        return event_data


    def game1_cv_test(self, ifold, odd_phi, even_phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2 & decision &
        trial_corr, accuracy = self.label_trial_corr()
        trial_label = self.split_trials(trial_corr)
        m2ev_corr_odd, m2ev_corr_even, m2ev_error = self.genM2ev(trial_label)
        deev_corr_odd, deev_corr_even, deev_error = self.genDeev(trial_label)
        # pmod
        pmod_alignPhi_odd = self.genpm_test(m2ev_corr_odd, ifold, even_phi, 'odd')
        pmod_alignPhi_even = self.genpm_test(m2ev_corr_even, ifold, odd_phi, 'even')

        event_data = pd.concat([m1ev, m2ev_corr_odd, m2ev_corr_even, m2ev_error,
                                deev_corr_odd, deev_corr_even, deev_error,
                                pmod_alignPhi_odd, pmod_alignPhi_even], axis=0)
        return event_data


    def game1_cv_test2(self, ifold, odd_phi, even_phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2
        trial_corr, accuracy = self.label_trial_corr()
        trial_label = self.split_trials(trial_corr, 'whole_trials')
        m2ev_odd, m2ev_even, _ = self.genM2ev_whole_trials(trial_label)
        # decision
        decision_corr, decision_error = self.genDeev_without_pmod()
        # pmod
        pmod_alignPhi_odd = self.genpm_test(m2ev_odd, ifold, even_phi, 'odd')
        pmod_alignPhi_even = self.genpm_test(m2ev_even, ifold, odd_phi, 'even')
        # pressButton
        pressButton = self.response()
        event_data = pd.concat([m1ev, m2ev_odd, m2ev_even, decision_corr, decision_error, pressButton,
                                pmod_alignPhi_odd, pmod_alignPhi_even], axis=0)
        return event_data

    def game1_cv_test3(self, ifold, odd_phi, even_phi):
        # using whole trials
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()  # m1
        # m2
        trial_corr, accuracy = self.label_trial_corr()
        trial_label = self.split_trials(trial_corr, 'whole_trials')
        m2ev_odd, m2ev_even, _ = self.genM2ev_whole_trials(trial_label)
        # decision
        deev_odd, deev_even, deev_error = self.genDeev_whole_trials(trial_corr, trial_label)
        # pmod
        pmod_alignPhi_m2_odd = self.genpm_test(m2ev_odd, ifold, even_phi, 'odd')
        pmod_alignPhi_m2_even = self.genpm_test(m2ev_even, ifold, odd_phi, 'even')

        pmod_alignPhi_decision_odd = self.genpm_test(deev_odd, ifold, even_phi, 'odd')
        pmod_alignPhi_decision_even = self.genpm_test(deev_even, ifold, odd_phi, 'even')

        event_data = pd.concat([m1ev, m2ev_odd, m2ev_even, deev_odd, deev_even,deev_error,
                                pmod_alignPhi_m2_odd, pmod_alignPhi_m2_even,
                                pmod_alignPhi_decision_odd,pmod_alignPhi_decision_even], axis=0)
        return event_data


def gen_event_game1_cv_train():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query("game1_fmri>=0.5")
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/cv_train1/sub-{}/{}fold'
    event_file = 'sub-{}_task-game1_run-{}_events.tsv'

    # set folds and runs for cross validation
    ifolds = range(6, 7)
    runs = range(1, 7)

    for sub in subject_list:
        print(sub, "started.")
        for ifold in ifolds:
            for run_id in runs:
                # generate event
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv(behDataPath)
                event = game1_cv.game1_cv_train_M2(ifold)
                # save
                out_dir = save_dir.format(sub, ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path = join(out_dir, event_file.format(sub, run_id))
                event.to_csv(tsv_save_path, sep="\t", index=False)


def gen_event_game1_cv_test():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query("game1_fmri>=0.5")
    data = data.query("(game1_acc>=0.80)and(Age>=18)")
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/cv_test1_RSA-EC_weighted_average/sub-{}/{}fold'  # look out
    event_file = 'sub-{}_task-game1_run-{}_events.tsv'

    # set Phi estimated from specific ROI
    odd_phi_file = r'/mnt/workdir/DCM/result/CV/Phi/2023.1.2/correct_trials/estPhi_ROI-RSA-ECthr3.1_On-M2_trials-odd_subjects-hp_weighted_average.csv'  # look out
    odd_phi_data = pd.read_csv(odd_phi_file)

    even_phi_file = r'/mnt/workdir/DCM/result/CV/Phi/2023.1.2/correct_trials/estPhi_ROI-RSA-ECthr3.1_On-M2_trials-even_subjects-hp_weighted_average.csv'  # look out
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
                event = game1_cv.game1_cv_test(ifold, odd_phi, even_phi)  # look out
                # save
                out_dir = save_dir.format(sub, ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path = join(out_dir, event_file.format(sub, run_id))
                event.to_csv(tsv_save_path, sep="\t", index=False)


def Test_gen_event_game1_cv_train():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]
    subject_list = random.sample(subject_list, 3)
    pass


#gen_event_game1_cv_train()
gen_event_game1_cv_test()