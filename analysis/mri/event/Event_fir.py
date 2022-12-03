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


class Game1EV(object):
    """"""
    def __init__(self, behDataPath):
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.behData = self.behData.fillna('None')
        self.dformat = None

    def game1_dformat(self):
        columns = self.behData.columns
        if 'fix_start_cue.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            print("The data is not game1 behavioral data.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fix_start_cue.started'][1]
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min() - 1
        else:
            raise Exception("Error:You need specify behavioral data format.")
        return starttime

    def label_trial_corr(self):
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
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData), 3)
        return trial_corr, accuracy

    def inferev(self, trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            angle = self.behData['angles']
            inferev = pd.DataFrame({'onset': onset, 'angle': angle})
            inferev['trial_type'] = 'inference'
            inferev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            angle = self.behData['angles']
            inferev = pd.DataFrame({'onset': onset, 'angle': angle})
            inferev['trial_type'] = 'inference'
            inferev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        infer_corr = pd.DataFrame(columns=['onset',  'angle'])
        infer_error = pd.DataFrame(columns=['onset', 'angle'])
        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                infer_corr = infer_corr.append(inferev.iloc[i])
            elif trial_label == False:
                infer_error = infer_error.append(inferev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        infer_corr['trial_type'] = 'infer_corr'
        infer_error['trial_type'] = 'infer_error'
        return infer_corr, infer_error

    def hexagon_pm(self, infer_corr, ifold):
        angle = infer_corr['angle']
        pmod_sin = infer_corr.copy()
        pmod_cos = infer_corr.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold * angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold * angle))
        return pmod_sin, pmod_cos

    def game1ev(self, ifold):
        self.starttime = self.cal_start_time()
        trial_corr, accuracy = self.label_trial_corr()
        infer_corr, infer_error = self.inferev(trial_corr)
        pmod_sin, pmod_cos = self.hexagon_pm(infer_corr, ifold)

        event_data = pd.concat([infer_corr, infer_error,
                                pmod_sin, pmod_cos], axis=0)
        return event_data


def gen_sub_event(task, subjects):
    if task == 'game1':
        runs = range(1, 7)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/fir_hexagon/sub-{}/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    elif task == 'game2':
        runs = range(1, 3)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1-test/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/{}/fir_hexagon/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")
    ifolds = range(6, 7)

    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))

        for ifold in ifolds:
            save_dir = template['save_dir'].format(task,subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in runs:
                run_id = str(idx)
                behDataPath = template['behav_path'].format(subj,subj,task,run_id)
                if task == 'game1':
                    event = Game1EV(behDataPath)
                    event_data = event.game1ev(ifold)
                else:
                    raise Exception("The type of task is wrong.")
                tsv_save_path = join(save_dir,template['event_file'].format(subj,task,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    task = 'game1'
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]
    gen_sub_event(task,subjects)