#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:28:44 2022

@author: dell
"""

import os
from os.path import join
import numpy as np
import pandas as pd


class Game1_alignPhi(object):
    def __init__(self, behDataPath):
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.dformat = None

    def game1_dformat(self):
        columns = self.behData.columns
        if 'fix_start_cue.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            raise Exception("You need specify behavioral data format.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fix_start_cue.started'][1]
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min() - 1
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime

    def label_trial_corr(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['resp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['resp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            raise Exception("You need specify behavioral data format.")

        trial_corr = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                else:
                    correctAns = 2
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                else:
                    correctAns = 1
            else:
                raise Exception("None of rule have been found in the file.")
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData), 3)
        return trial_corr, accuracy

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic1_render.started'] - self.starttime
            duration = self.behData['pic2_render.started'] - self.behData['pic1_render.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic1_render.started_raw'] - self.starttime
            duration = self.behData['pic2_render.started_raw'] - self.behData['pic1_render.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset', ignore_index=True)
        return m1ev

    def genM2ev(self, trial_corr):
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

        m2ev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        m2ev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(m2ev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."

        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
            elif trial_label == False:
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr, m2ev_error

    def genDeev(self, trial_corr):
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

        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif trial_label == False:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_corr, deev_error

    def pressButton(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        pbev = pbev.sort_values('onset', ignore_index=True)
        return pbev

    def genpm(self, m2ev_corr, ifold, phi):
        angle = m2ev_corr['angle']
        pmod_alignPhi = m2ev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game1_alignPhi(self, ifold, phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        pbev = self.pressButton()
        trial_corr, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_alignPhi = self.genpm(m2ev_corr, ifold, phi)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error, pbev,
                                pmod_alignPhi], axis=0)
        return event_data


class Game2AlignPhi(object):
    def __init__(self, behDataPath):
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.dformat = None

    def game2_dformat(self):
        columns = self.behData.columns
        if 'fixation.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            print("The data is not game2 behavioral data.")

    def cal_start_time(self):
        self.game2_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fixation.started'].min()
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min()
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime

    def label_trial_corr(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['dResp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['dResp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            raise Exception("You need specify behavioral data format.")

        trial_corr = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                else:
                    correctAns = 2
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                else:
                    correctAns = 1
            else:
                raise Exception("None of rule have been found in the file.")
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData), 3)
        return trial_corr, accuracy

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic1.started'] - self.starttime
            duration = self.behData['testPic2.started'] - self.behData['testPic1.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic1.started_raw'] - self.starttime
            duration = self.behData['testPic2.started_raw'] - self.behData['testPic1.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset', ignore_index=True)
        return m1ev

    def genM2ev(self, trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        m2ev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        m2ev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])

        assert len(m2ev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."

        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
            elif trial_label == False:
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr, m2ev_error

    def genDeev(self, trial_corr):
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

        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif trial_label == False:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_corr, deev_error

    def genpm(self, m2ev_corr, ifold, phi):
        angle = m2ev_corr['angle']
        pmod_alignPhi = m2ev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game2_alignPhi(self, ifold, phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_alignPhi = self.genpm(m2ev_corr, ifold, phi)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_alignPhi], axis=0)
        return event_data


def game2_align_game1_event():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game2_fmri>0.5')
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    phi_file = r'/mnt/workdir/DCM/result/CV/Phi/2022.12.21/estPhi_ROI-bigmPFC_On-M2_trial-corr_subject-all.csv'
    ifolds = range(6, 7)

    phi_data = pd.read_csv(phi_file)
    for subj in subject_list:
        print('----sub-{}----'.format(subj))
        behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-game2_run-{}.csv'
        savedir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/alignPhi_mPFC/sub-{}/{}fold'  # need change
        event_file = 'sub-{}_task-game2_run-{}_events.tsv'
        for ifold in ifolds:
            sub_ifold_phi = phi_data.query(f'(sub_id=="sub-{subj}")and(ifold=="{ifold}fold")')['Phi'].values[0]
            savepath = savedir.format(subj, ifold)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            for idx in range(1, 3):
                run_id = str(idx)
                behDataPath = behav_path.format(subj, subj, run_id)
                event = Game2AlignPhi(behDataPath)
                event_data = event.game2_alignPhi(ifold, sub_ifold_phi)
                save_path = join(savepath, event_file.format(subj, run_id))
                event_data.to_csv(save_path, sep="\t", index=False)

game2_align_game1_event()
