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
            print("You need specify behavioral data format.")
        return m1ev
    
    
    def genM2ev(self,trial_corr,error_trial=True):

        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            #duration = self.behData['cue1.started'] - self.behData['pic2_render.started']
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            #duration = self.behData['cue1.started_raw'] - self.behData['pic2_render.started_raw']
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'  
            m2ev['modulation'] = 1
        else:
            print("You need specify behavioral data format.")
        
        if error_trial:
            m2ev_corr = pd.DataFrame(columns=['onset','duration','angle'])
            m2ev_error = pd.DataFrame(columns=['onset','duration','angle'])
            for i,trial_label  in enumerate(trial_corr):
                if trial_label == True:
                    m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
                elif trial_label == False:
                    m2ev_error = m2ev_error.append(m2ev.iloc[i])
                else:
                    raise ValueError("The trial label should be True or False.")
            m2ev_corr['trial_type'] = 'M2_corr'
            m2ev_error['trial_type'] = 'M2_error'
            return m2ev_corr, m2ev_error
        else:
            return m2ev
 
    
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
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData),3)        
        return trial_corr,accuracy
    
    
    def M2pm(self,m2ev,trial_corr,ifold):
        if isinstance(m2ev, tuple):
            m2ev, _ = m2ev # pmod on correct trials
        
        angle = m2ev['angle']            
        pmod_sin = m2ev.copy()
        pmod_cos = m2ev.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
        
        return pmod_sin, pmod_cos


    def genDeev(self):
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
            print("You need specify behavioral data format.")
        return deev


    def game1ev_hexonM2(self,ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        deev = self.genDeev()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev = self.genM2ev(trial_corr,True)
        pmod_sin, pmod_cos = self.M2pm(m2ev, trial_corr, ifold)
        if isinstance(m2ev, tuple):
            m2ev_corr, m2ev_error = m2ev 
            
        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,
                                pmod_sin,pmod_cos,deev],axis=0)
        return event_data
    

if __name__ == "__main__":
    """
    subjects = ['033', '048', '049', '050', '055', '056', '058',
                '059', '060', '061', '062', '063', '064']
    participants_tsv = r'/mnt/data/Project/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query('usable==1')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('_')[-1] for p in pid]
    """
    subjects = ['065']
    runs = range(1,7)
    ifolds = range(4,9)
    
    template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv',
                'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/hexonM2short/{}fold',
                'event_file':'sub-{}_task-game1_run-{}_events.tsv'}
    
    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))
        
        for ifold in ifolds:
            save_dir = template['save_dir'].format(subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            for idx in runs:
                run_id = str(idx)
                behDataPath = template['behav_path'].format(subj,subj,run_id)
                event = Game1EV(behDataPath)
                event_data = event.game1ev_hexonM2(ifold)
                tsv_save_path = join(save_dir,template['event_file'].format(subj,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)