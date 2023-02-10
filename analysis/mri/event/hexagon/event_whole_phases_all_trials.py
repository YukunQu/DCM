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
    """
    model the hexagonal effect from the onset of M2 to press button,
    but also model the onset time of visual stimuli with stick function.
    """
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
            print("The data is not game1 behavioral data.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fix_start_cue.started'].min()
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min() - 1
        else:
            raise Exception("The input file is wrong! You need specify behavioral data format.")
        return starttime

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


    def inferev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['pic2_render.started']
            angle = self.behData['angles']

            inferev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            inferev['trial_type'] = 'inference'
            inferev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['pic2_render.started_raw']
            angle = self.behData['angles']

            inferev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            inferev['trial_type'] = 'inference'
            inferev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        return inferev

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
        m1ev =self.genM1ev()
        inferev = self.inferev()
        pmod_sin, pmod_cos = self.hexagon_pm(inferev, ifold)
        event_data = pd.concat([m1ev,inferev,pmod_sin, pmod_cos], axis=0)
        return event_data

def gen_sub_event(task, subjects):
    if task == 'game1':
        runs = range(1,7)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/whole_hexagon_all_trials/sub-{}/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    elif task == 'game2':
        runs = range(1,3)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/whole_hexagon_all_trials/sub-{}/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")

    ifolds = range(4,9)

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
                elif task == 'game2':
                    continue
                    #event = Game2EV(behDataPath)
                    #event_data = event.game2ev(ifold)
                else:
                    raise Exception("The type of task is wrong.")
                tsv_save_path = join(save_dir,template['event_file'].format(subj,task,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)

if __name__ == "__main__":

    task = 'game1'
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query(f'{task}_fmri==1')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]
    gen_sub_event(task,subjects)