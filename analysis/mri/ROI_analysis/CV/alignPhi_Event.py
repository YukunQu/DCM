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
    def __init__(self,behDataPath):
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
        return m1ev

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
            else:
                raise ValueError("The trial label should be True or False.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'
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
        return deev_corr, deev_error

    def pressButton(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['resp.started'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['resp.started_raw'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        return pbev

    def genpm(self,m2ev_corr,ifold,phi):
        angle = m2ev_corr['angle']
        pmod_alignPhi = m2ev_corr.copy()
        pmod_alignPhi['trial_type'] = 'alignPhi'
        pmod_alignPhi['modulation'] = np.cos(np.deg2rad(ifold * (angle - phi)))
        return pmod_alignPhi

    def game1_alignPhi(self, ifold, phi):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev_corr,m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pbev = self.pressButton()
        pmod_alignPhi = self.genpm(m2ev_corr,ifold,phi)
        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,deev_corr,deev_error,pbev,
                                pmod_alignPhi],axis=0)
        return event_data


if __name__ == "__main__":
    task = 'game1'
    glm_type = 'separate_hexagon'
    # define roi
    roi_type = 'park'
    phi_type = 'ec_phi'  # look out ec_phi or vmpfc_phi

    if roi_type == 'func':
        if phi_type == 'ec_phi':
            save_containter = 'game1/alignPhi/EC_func'
        elif phi_type == 'vmpfc_phi':
            save_containter = 'game1/alignPhi/vmpfc_func'
        else:
            raise Exception("phi type is wrong.")
    elif roi_type == 'park':
        if phi_type == 'ec_phi':
            save_containter = 'game1/alignPhi/EC_park'
        elif phi_type == 'vmpfc_phi':
            save_containter = 'game1/alignPhi/EC_vmpfc'
        else:
            raise Exception("phi type is wrong.")
    else:
        raise Exception("roi type is wrong.")

    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    data = data.query('game1_acc>=0.8')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('_')[-1] for p in pid]

    # default setting
    ifolds = range(4, 9)
    template = {
        'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv',
        'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/'+save_containter+'/testset{}/{}fold',
        'event_file': 'sub-{}_task-game1_run-{}_events.tsv'}

    # define test set
    test_configs = {'1': [4, 5, 6],
                    '2': [1, 2, 3]}

    for test_id, test_runs in test_configs.items():
        phi_data = pd.read_csv(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/{}/Phi'
                               r'/estPhi_{}_Set{}_{}_ROI.csv'.format(task,glm_type,task,test_id,roi_type)) # look out ROI
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