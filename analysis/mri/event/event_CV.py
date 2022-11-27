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


class Game1_cv(object):
    def __init__(self,behDataPath):
        self.starttime = None
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
        for keyResp,row in zip(keyResp_list, self.behData.itertuples()):
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
        accuracy = np.round(np.sum(trial_corr) / len(self.behData),3)
        return trial_corr,accuracy

    def _split_corr_trials(self,trial_label):
        i = 0
        odd_trial = []
        even_trial = []
        for label in trial_label:
            if label:
                i+=1
                if i%2==0:
                    odd_trial.append(None)
                    even_trial.append(label)
                else:
                    odd_trial.append(label)
                    even_trial.append(None)
            else:
                odd_trial.append(label)
                even_trial.append(label)
        return odd_trial,even_trial

    def _split_whole_trials(self,trial_label):
        i = 0
        odd_trial = []
        even_trial = []
        for label in trial_label:
            i += 1
            if i%2==0:
                odd_trial.append(None)
                even_trial.append(label)
            else:
                odd_trial.append(label)
                even_trial.append(None)
        return odd_trial,even_trial

    def split_trials(self,trial_label,trial_type,split_target='corr_trials'):
        if split_target == 'corr_trials':
            odd_trial,even_trial = self._split_corr_trials(trial_label)
        elif split_target == 'whole_trials':
            odd_trial,even_trial = self._split_whole_trials(trial_label)
        else:
            raise Exception("The split target is not supported.")

        if trial_type == 'odd':
            return odd_trial
        elif trial_type == 'even':
            return even_trial
        elif trial_label == 'both':
            return odd_trial,even_trial

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic1_render.started'] - self.starttime
            duration = self.behData['pic2_render.started'] - self.behData['pic1_render.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic1_render.started_raw'] - self.starttime
            duration = self.behData['pic2_render.started_raw'] - self.behData['pic1_render.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset',ignore_index=True)
        return m1ev

    def genM2ev(self,trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        m2ev_corr = pd.DataFrame(columns=['onset','duration','angle'])
        m2ev_error = pd.DataFrame(columns=['onset','duration','angle'])

        assert len(m2ev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."

        for i,trial_label in enumerate(trial_corr):
            if trial_label == True:
                m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
            elif trial_label == False:
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            elif trial_label == None:
                continue
            else:
                raise ValueError("The trial label should be True,False or None.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr, m2ev_error

    def genDeev(self,trial_corr):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        deev_corr = pd.DataFrame(columns=['onset','duration','angle'])
        deev_error = pd.DataFrame(columns=['onset','duration','angle'])

        assert len(deev) == len(trial_corr), "The number of trial label didn't not  same as the number of event-decision."

        for i,trial_label in enumerate(trial_corr):
            if trial_label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif trial_label == False:
                deev_error = deev_error.append(deev.iloc[i])
            elif trial_label == None:
                continue
            else:
                raise ValueError("The trial label should be True,False or None.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset',ignore_index=True)
        deev_error = deev_error.sort_values('onset',ignore_index=True)
        return deev_corr, deev_error

    def pressButton(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        pbev = pbev.sort_values('onset',ignore_index=True)
        return pbev

    def genpm_train(self,m2ev_corr,ifold):
        angle = m2ev_corr['angle']
        pmod_sin = m2ev_corr.copy()
        pmod_cos = m2ev_corr.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
        return pmod_sin, pmod_cos

    def genpm_test(self,m2ev_corr,ifold,phi):
        angle = m2ev_corr['angle']
        pmod_alignPhi = m2ev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game1_cv_train(self,ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        pbev = self.pressButton()
        trial_label,accuracy = self.label_trial_corr()
        odd_trial,even_trial = self.split_trials(trial_label,'both')

        # even event
        m2ev_corr, m2ev_error = self.genM2ev(even_trial)
        deev_corr, deev_error = self.genDeev(even_trial)
        pmod_sin, pmod_cos = self.genpm_train(m2ev_corr,ifold)

        even_event = pd.concat([m1ev,m2ev_corr,m2ev_error,deev_corr,deev_error,pbev,
                                pmod_sin,pmod_cos],axis=0)

        # odd event
        m2ev_corr, m2ev_error = self.genM2ev(odd_trial)
        deev_corr, deev_error = self.genDeev(odd_trial)
        pmod_sin, pmod_cos = self.genpm_train(m2ev_corr,ifold)

        odd_event = pd.concat([m1ev,m2ev_corr,m2ev_error,deev_corr,deev_error,pbev,
                            pmod_sin,pmod_cos],axis=0)
        return even_event,odd_event

    def game1_cv_test(self, ifold, phi, trial_type):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_label,accuracy = self.label_trial_corr()
        trial_set = self.split_trials(trial_label,trial_type)

        # generate event of trials
        m2ev_corr, m2ev_error = self.genM2ev(trial_set)
        deev_corr, deev_error = self.genDeev(trial_set)
        pmod_alignPhi = self.genpm_test(m2ev_corr,ifold,phi)
        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,deev_corr,deev_error,pmod_alignPhi],axis=0)
        return event_data


class Game2_alignPhi(object):
    def __init__(self,behDataPath):
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
        for keyResp,row in zip(keyResp_list, self.behData.itertuples()):
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
        accuracy = np.round(np.sum(trial_corr) / len(self.behData),3)
        return trial_corr,accuracy

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic1.started'] - self.starttime
            duration = self.behData['testPic2.started'] - self.behData['testPic1.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic1.started_raw'] - self.starttime
            duration = self.behData['testPic2.started_raw'] - self.behData['testPic1.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset',ignore_index=True)
        return m1ev

    def genM2ev(self,trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        m2ev_corr = pd.DataFrame(columns=['onset','duration','angle'])
        m2ev_error = pd.DataFrame(columns=['onset','duration','angle'])

        assert len(m2ev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."

        for i,trial_label in enumerate(trial_corr):
            if trial_label == True:
                m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
            elif trial_label == False:
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset',ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset',ignore_index=True)
        return m2ev_corr, m2ev_error

    def genDeev(self,trial_corr):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        deev_corr = pd.DataFrame(columns=['onset','duration','angle'])
        deev_error = pd.DataFrame(columns=['onset','duration','angle'])

        assert len(deev) == len(trial_corr), "The number of trial label didn't not  same as the number of event-decision."

        for i,trial_label in enumerate(trial_corr):
            if trial_label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif trial_label == False:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset',ignore_index=True)
        deev_error = deev_error.sort_values('onset',ignore_index=True)
        return deev_corr, deev_error

    def pressButton(self):
        if self.dformat == 'trial_by_trial':
            pressB_data = self.behData.copy()
            pressB_data = pressB_data.dropna(axis=0,subset=['dResp.rt'])
            onset = pressB_data['cue1.started'] - self.starttime + pressB_data['dResp.rt']
            duration = 0
            angle = pressB_data['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            pressB_data = self.behData.copy().dropna(axis=0, subset=['dResp.rt_raw'])
            onset = pressB_data['cue1.started_raw'] - self.starttime + pressB_data['dResp.rt_raw']
            duration = 0
            angle = pressB_data['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        pbev = pbev.sort_values('onset',ignore_index=True)
        return pbev

    def genpm(self,m2ev_corr,ifold,phi):
        angle = m2ev_corr['angle']
        pmod_alignPhi = m2ev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game2_alignPhi(self, ifold, phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        pbev = self.pressButton()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev_corr,m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_alignPhi = self.genpm(m2ev_corr,ifold,phi)
        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,deev_corr,deev_error,pbev,
                                pmod_alignPhi],axis=0)
        return event_data


def game2_align_game1_event(phi_file,subject_list,ifolds,phi_type,roi_type):
    phi_data = pd.read_csv(phi_file)
    for subj in subject_list:
        print('----sub-{}----'.format(subj))
        behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-game2_run-{}.csv'
        savedir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/game2/alignPhi/{}/{}fold'  # need change
        event_file = 'sub-{}_task-game2_run-{}_events.tsv'
        for ifold in ifolds:
            sub_ifold_phi = phi_data.query(f'(sub_id=="sub-{subj}")and(ifold=="{ifold}fold")')[phi_type].values[0]
            savepath = savedir.format(subj,roi_type,ifold)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            for idx in range(1,3):
                run_id = str(idx)
                behDataPath = behav_path.format(subj, subj,run_id)
                event = Game2_alignPhi(behDataPath)
                event_data = event.game2_alignPhi(ifold, sub_ifold_phi)
                save_path = join(savepath, event_file.format(subj, run_id))
                event_data.to_csv(save_path, sep="\t", index=False)


def gen_event_game1_cv_train():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/separate_hexagon_correct_trials_train/sub-{}/{}fold'
    event_file = 'sub-{}_task-game1_run-{}_set-{}_events.tsv'

    # set folds and runs for cross validation
    ifolds = range(4, 9)
    runs = range(1,7)

    for sub in subject_list:
        print(sub,"started.")
        for ifold in ifolds:
            for run_id in runs:
                # generate event
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv(behDataPath)
                even_event,odd_event = game1_cv.game1_cv_train(ifold)
                # save
                out_dir = save_dir.format(sub,ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path1 = join(out_dir,event_file.format(sub,run_id,'even'))
                tsv_save_path2 = join(out_dir,event_file.format(sub,run_id,'odd'))
                even_event.to_csv(tsv_save_path1, sep="\t", index=False)
                odd_event.to_csv(tsv_save_path2, sep="\t", index=False)


def gen_event_game1_cv_test():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    # define the template of behavioral file
    behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/' \
               r'separate_hexagon_correct_trials_test_ROI-EC/sub-{}/{}fold'  # look out
    event_file = 'sub-{}_task-game1_run-{}_set-{}_events.tsv'

    # set Phi estimated from specific ROI
    odd_phi_file = r'/mnt/workdir/DCM/result/CV/Phi/estPhi_ROI-EC_On-average_Set2.csv'  # look out
    odd_phi_data = pd.read_csv(odd_phi_file)

    even_phi_file = r'/mnt/workdir/DCM/result/CV/Phi/estPhi_ROI-EC_On-average_Set1.csv'  # look out
    even_phi_data = pd.read_csv(even_phi_file)

    # set folds and runs for cross validation
    ifolds = range(6, 7)
    runs = range(1,7)

    for sub in subject_list:
        print(sub,"started.")
        for ifold in ifolds:
            odd_phi = odd_phi_data.query(f'(sub_id=="sub-{sub}")and(ifold=="{ifold}fold")')['Phi'].values[0]
            even_phi = even_phi_data.query(f'(sub_id=="sub-{sub}")and(ifold=="{ifold}fold")')['Phi'].values[0]
            for run_id in runs:
                behDataPath = behav_path.format(sub, sub, run_id)
                game1_cv = Game1_cv(behDataPath)
                odd_event = game1_cv.game1_cv_test(ifold, even_phi, 'odd')
                even_event = game1_cv.game1_cv_test(ifold, odd_phi, 'even')
                # save
                out_dir = save_dir.format(sub,ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path = join(out_dir,event_file.format(sub,run_id,'odd'))
                odd_event.to_csv(tsv_save_path, sep="\t", index=False)
                tsv_save_path = join(out_dir,event_file.format(sub,run_id,'even'))
                even_event.to_csv(tsv_save_path, sep='\t', index=False)


def Test_gen_event_game1_cv_train():
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]
    subject_list = random.sample(subject_list,3)
    pass


gen_event_game1_cv_test()