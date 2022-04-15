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


class AlignPhi(object):
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
            print("Error:You need specify behavioral data format.")
        return starttime

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
            print("You need specify behavioral data format.")
        return m1ev

    def genM2ev(self):
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
            print("You need specify behavioral data format.")
        return m2ev

    def genDeev(self):
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
            print("You need specify behavioral data format.")
        return deev

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
            print("You need specify behavioral data format.")

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

    def hexmodev(self, trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['pic2_render.started']
            angle = self.behData['angles']
            hexmodev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            hexmodev['trial_type'] = 'hexmod'
            hexmodev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['pic2_render.started_raw']
            angle = self.behData['angles']
            hexmodev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            hexmodev['trial_type'] = 'hexmod'
            hexmodev['modulation'] = 1
        else:
            print("You need specify behavioral data format.")

        hexev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        hexev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                hexev_corr = hexev_corr.append(hexmodev.iloc[i])
            elif trial_label == False:
                hexev_error = hexev_error.append(hexmodev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        hexev_corr['trial_type'] = 'hex_corr'
        hexev_error['trial_type'] = 'hex_error'
        return hexev_corr, hexev_error

    def hexpm(self, hexev_corr, ifold, phi):
        angle = hexev_corr['angle']
        pmod_alignPhi = hexev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game1_alignPhi(self, ifold, phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        trial_corr, accuracy = self.label_trial_corr()
        hexev_corr, hexev_error = self.hexmodev(trial_corr)
        pmod_alignPhi = self.hexpm(hexev_corr, ifold, phi)

        event_data = pd.concat([m1ev, m2ev, deev,
                                hexev_corr, hexev_error,
                                pmod_alignPhi], axis=0)
        return event_data


if __name__ == "__main__":
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('(usable==1)&(game1_acc>0.75)&(Age>18)')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('_')[-1] for p in pid]

    # define roi
    phi_type = 'ec_phi'  # look out ec_phi or vmpfc_phi

    # default setting
    ifolds = range(4, 9)
    template = {
        'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv',
        'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/alignPhi/EC/testset{}/{}fold', # look out ROI
        'event_file': 'sub-{}_task-game1_run-{}_events.tsv'}

    # define test set
    test_configs = {'1': [4, 5, 6],
                    '2': [1, 2, 3]}
    #test_configs = {'all': [1, 2, 3, 4, 5, 6]}

    for test_id, test_runs in test_configs.items():
        phi_data = pd.read_csv(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/'
                               r'specificTo6/Phi/trainset{}_estPhi_individual_ROI.csv'.format(test_id)) # look out ROI
        for subj in subjects:
            print('----sub-{}----'.format(subj))

            for ifold in ifolds:
                sub_ifold_phi = phi_data.query(f'(sub_id=="sub-{subj}")and(ifold=="{ifold}fold")')[phi_type].values[0]
                save_dir = template['save_dir'].format(subj,test_id,ifold)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                for idx in test_runs:
                    run_id = str(idx)
                    behDataPath = template['behav_path'].format(subj, subj, run_id)
                    event = AlignPhi(behDataPath)
                    event_data = event.game1_alignPhi(ifold, sub_ifold_phi)
                    save_path = join(save_dir, template['event_file'].format(subj, run_id))
                    event_data.to_csv(save_path, sep="\t", index=False)