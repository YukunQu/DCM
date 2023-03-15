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
from analysis.mri.event.distance.event_distance import Game1EV_hexagon_distance_spct,Game1EV_hexagon_distance_spat


class Game1_cv_spat(Game1EV_hexagon_distance_spat):
    def __init__(self, behDataPath):
        Game1EV_hexagon_distance_spat.__init__(self, behDataPath)

    @staticmethod
    def split_whole_trials(trial_corr):
        trial_label = []
        for i in range(1, len(trial_corr) + 1):
            if i % 2 == 0:
                trial_label.append('even')
            else:
                trial_label.append('odd')
        return trial_label

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


class Game1_cv_spct(Game1EV_hexagon_distance_spct):
    def __init__(self, behDataPath):
        Game1EV_hexagon_distance_spct.__init__(self, behDataPath)

    @staticmethod
    def split_event(ev,trial_corr):
        # split correct trials into odd trials and even trials
        i = 0
        trial_label = []
        for tc in trial_corr:
            # If the trial is correct tiral, it will be labeld.
            if tc:
                i += 1
                if i % 2 != 0:
                    trial_label.append('odd')
                else:
                    trial_label.append('even')
            else:
                continue

        odd_trials_index = []
        even_trials_index = []
        for i, label in enumerate(trial_label):
            if label == 'odd':
                odd_trials_index.append(i)
            elif label == 'even':
                even_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        odd_ev = ev.iloc[odd_trials_index].copy()
        even_ev = ev.iloc[even_trials_index].copy()
        return odd_ev,even_ev

    def genpm_train(self, ev, ifold, trial_type):
        # generate parametric modulation for training GLM
        angle = ev['angle']
        pmod_sin = ev.copy()
        pmod_cos = ev.copy()
        pmod_sin['modulation'] = np.round(np.sin(np.deg2rad(ifold * angle)),2)
        pmod_cos['modulation'] = np.round(np.cos(np.deg2rad(ifold * angle)),2)
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
        # generate parametric modulation for test GLM
        angle = ev['angle']
        pmod_alignPhi = ev.copy()
        pmod_alignPhi['modulation'] = np.round(np.cos(np.deg2rad(ifold * (angle - phi))),2)
        if trial_type == 'odd':
            pmod_alignPhi['trial_type'] = 'alignPhi_odd'
        elif trial_type == 'even':
            pmod_alignPhi['trial_type'] = 'alignPhi_even'
        else:
            raise Exception("The variable 'trial type' only support odd and 'even'")
        return pmod_alignPhi

    def game1_cv_train(self, ifold):
        # generatea M1, M2, Decision's event
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        # split trial into correct trial and error trial
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)

        # split correct trials into odd trials and even trials
        m2ev_odd,m2ev_even = self.split_event(m2ev_corr,trial_corr)
        deev_odd,deev_even = self.split_event(deev_corr,trial_corr)

        # generate pmod of odd trials and even trials
        m2_sin_odd, m2_cos_odd = self.genpm_train(m2ev_odd, ifold, 'odd')
        decision_sin_odd, decision_cos_odd = self.genpm_train(deev_odd, ifold, 'odd')
        sin_odd = pd.concat([m2_sin_odd,decision_sin_odd],axis=0).sort_values('onset', ignore_index=True)
        cos_odd = pd.concat([m2_cos_odd,decision_cos_odd],axis=0).sort_values('onset', ignore_index=True)

        m2_sin_even, m2_cos_even = self.genpm_train(m2ev_even, ifold, 'even')
        decision_sin_even, decision_cos_even = self.genpm_train(deev_even, ifold, 'even')
        sin_even = pd.concat([m2_sin_even,decision_sin_even],axis=0).sort_values('onset', ignore_index=True)
        cos_even = pd.concat([m2_cos_even,decision_cos_even],axis=0).sort_values('onset', ignore_index=True)

        # generate pmod of distance
        m2ev_distance = self.genpm_distance_spct(trial_corr)
        m2ev_distance['trial_type'] = 'M2_corrx' + m2ev_distance['trial_type']
        distance_pm = m2ev_distance['modulation']
        deev_distance = deev_corr.copy()
        deev_distance['modulation'] = distance_pm
        deev_distance['trial_type'] = deev_distance['trial_type'] + 'xdistance'

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                sin_odd,cos_odd,sin_even,cos_even,
                                m2ev_distance,deev_distance],axis=0)
        return event_data

    def game1_cv_test(self, ifold, odd_phi, even_phi):
        # generatea M1, M2, Decision's event
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        # split trial into correct trial and error trial
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)

        # split correct trials into odd trials and even trials
        m2ev_odd,m2ev_even = self.split_event(m2ev_corr,trial_corr)
        deev_odd,deev_even = self.split_event(deev_corr,trial_corr)

        # generate pmod of odd trials and even trials
        # odd trials
        m2_alignPhi_odd = self.genpm_test(m2ev_odd, ifold, even_phi, 'odd')
        decision_alignPhi_odd = self.genpm_test(deev_odd, ifold, even_phi, 'odd')
        alignPhi_odd = pd.concat([m2_alignPhi_odd,decision_alignPhi_odd],axis=0).sort_values('onset', ignore_index=True)

        # even trials
        m2_alignPhi_even = self.genpm_test(m2ev_even, ifold, odd_phi,'even')
        decision_alignPhi_even = self.genpm_test(deev_even, ifold, odd_phi,'even')
        alignPhi_even = pd.concat([m2_alignPhi_even,decision_alignPhi_even],axis=0).sort_values('onset', ignore_index=True)

        # generate pmod of distance
        m2ev_distance = self.genpm_distance_spct(trial_corr)
        m2ev_distance['trial_type'] = 'M2_corrx' + m2ev_distance['trial_type']
        distance_pm = m2ev_distance['modulation']
        deev_distance = deev_corr.copy()
        deev_distance['modulation'] = distance_pm
        deev_distance['trial_type'] = deev_distance['trial_type'] + 'xdistance'

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                alignPhi_odd,alignPhi_even,
                                m2ev_distance,deev_distance],axis=0)
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
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/cv_train_hexagon_distance_spct/sub-{}/{}fold'
    event_file = 'sub-{}_task-game1_run-{}_events.tsv'

    # set folds and runs for cross validation
    ifolds = range(4, 9)
    runs = range(1, 7)

    for sub in subject_list:
        print(sub, "started.")
        for ifold in ifolds:
            for run_id in runs:
                # generate event
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv_spct(behDataPath)
                event = game1_cv.game1_cv_train(ifold)
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
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    event_file = 'sub-{}_task-game1_run-{}_events.tsv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/cv_test_hexagon_distance_spct/sub-{}/{}fold'  # look out

    # set Phi estimated from specific ROI
    phis_file = r'/mnt/workdir/DCM/result/CV/Phi/estPhi_ROI-EC_circmean.csv'  # look out
    phis_data = pd.read_csv(phis_file)

    # set folds and runs for cross validation
    ifolds = range(4, 9)
    runs = range(1, 7)

    for ifold in ifolds:
        for sub in subject_list:
            print(ifold,'-',sub, "started.")
            odd_phi = phis_data.query(f'(sub_id=="sub-{sub}")and(ifold=="{ifold}fold")and(trial_type=="odd")')['Phi_mean'].values[0]
            even_phi = phis_data.query(f'(sub_id=="sub-{sub}")and(ifold=="{ifold}fold")and(trial_type=="even")')['Phi_mean'].values[0]
            for run_id in runs:
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv_spct(behDataPath)
                event = game1_cv.game1_cv_test(ifold, odd_phi, even_phi)  # look out
                # save
                out_dir = save_dir.format(sub, ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path = join(out_dir, event_file.format(sub, run_id))
                event.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    #gen_event_game1_cv_train()
    gen_event_game1_cv_test()
