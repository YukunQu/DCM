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
from analysis.mri.event.game1_event import Game1EV


class Game1EV_hexagon_wpct(Game1EV):
    """
    model the hexagonal effect from the onset from M2 to press button for correct trials
    """
    def __int__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)

    def gen_inferev(self, trial_corr):
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

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_corr):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        inferev_corr = inferev.iloc[correct_trials_index].copy()
        inferev_error = inferev.iloc[error_trials_index].copy()
        inferev_corr['trial_type'] = 'infer_corr'
        inferev_error['trial_type'] = 'infer_error'

        inferev_corr = inferev_corr.sort_values('onset', ignore_index=True)
        inferev_error = inferev_error.sort_values('onset', ignore_index=True)
        return inferev_corr, inferev_error

    def game1_hexagon_wpct_ev(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        infer_corr, infer_error = self.gen_inferev(trial_corr)
        pmod_sin, pmod_cos = self.genpm(infer_corr, ifold)

        event_data = pd.concat([m1ev,infer_corr,infer_error,
                                pmod_sin, pmod_cos], axis=0)
        return event_data


def gen_sub_event(task_type, subjects, ifolds=range(6, 7)):
    if task_type == 'game1':
        runs = range(1, 7)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'hexagon_wpct/sub-{}/{}fold',
                      'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    elif task_type == 'game2':
        runs = range(1, 3)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'hexagon_wpct/sub-{}/{}fold',
                      'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")

    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))

        for ifold in ifolds:
            save_dir = parameters['save_dir'].format(task_type, subj, ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in runs:
                run_id = str(idx)
                behDataPath = parameters['behav_path'].format(subj, subj, task_type, run_id)
                if task_type == 'game1':
                    event = Game1EV_hexagon_wpct(behDataPath)
                    event_data = event.game1_hexagon_wpct_ev(ifold)
                #elif task_type == 'game2':
                #    event = Game2EV_hexagon_spct(behDataPath)
                #    event_data = event.game2ev_hexagon_spct(ifold)
                else:
                    raise Exception("The type of task is wrong.")
                tsv_save_path = join(save_dir, parameters['event_file'].format(subj, task_type, run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    task = 'game1'
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects_list = [p.split('-')[-1] for p in pid]
    gen_sub_event(task, subjects_list)