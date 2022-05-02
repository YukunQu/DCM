# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:11:48 2021

@author: QYK
"""
import os
from os.path import join 
import numpy as np
import pandas as pd



# from the behavior data read the event parameter:condition;onsets;duration
def hexonM2(subj):
    # hexagon modulation effect when the second monster show
    """generate the event file of hexagon modulation"""
    runs = range(1,7)
    for ifold in range(4,9):
        for idx in runs:
            run_id = str(idx)
            # make directroy
            behav_data_path = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'.format(subj,subj,run_id)
            save_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events/sub-{}/hexonM2Long/{}fold'.format(subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)            

            # set start point
            behav_data = pd.read_csv(behav_data_path)
            behav_data = behav_data.dropna(axis=0, subset=['pairs_id'])
            start_time = behav_data['fix_start_cue.started'][1]
            
            # M1 event
            onset = behav_data['pic1_render.started'] - start_time
            duration = behav_data['pic2_render.started'] - behav_data['pic1_render.started']
            angle = behav_data['angles']
            tsv_data_m1 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})        
            tsv_data_m1['trial_type'] = 'M1'
            tsv_data_m1['modulation'] = 1
        
            # M2 event and hexagon modulation
            onset = behav_data['pic2_render.started'] - start_time
            duration = behav_data['cue1.started'] - behav_data['pic2_render.started']
            #duration = [2.5] * len(onset)
            angle = behav_data['angles']
            tsv_data_m2 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_m2['trial_type'] = 'M2'
            
            tsv_data_sin = tsv_data_m2.copy()
            tsv_data_cos = tsv_data_m2.copy()
            tsv_data_sin['trial_type'] = 'sin'
            tsv_data_cos['trial_type'] = 'cos'
        
            tsv_data_m2['modulation'] = 1
            tsv_data_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
            tsv_data_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
            
            # decision 
            onset =  behav_data['cue1.started'] - start_time
            duration = behav_data['cue1_2.started'] - behav_data['cue1.started']
            angle = behav_data['angles']
            tsv_data_rule = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_rule['trial_type'] = 'decision'
            tsv_data_rule['modulation'] = 1
            
            tsv_data = pd.concat([tsv_data_m1,tsv_data_m2,tsv_data_sin,tsv_data_cos,tsv_data_rule],axis=0)
            tsv_save_path = join(save_dir,'sub-{}_task-game1_run-{}_events.tsv'.format(subj,run_id))
            tsv_data.to_csv(tsv_save_path, sep="\t", index=False)


def hexonM2_v2(subj):
    # hexagon modulation effect when the second monster show
    """generate the event file of hexagon modulation"""
    runs = range(1,7)
    for ifold in range(4,9):
        for idx in runs:
            run_id = str(idx)
            
            # make directroy
            behav_data_path = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'.format(subj,subj,run_id)
            save_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events/sub-{}/hexonM2Long/{}fold'.format(subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)            

            # set start point
            behav_data = pd.read_csv(behav_data_path)
            behav_data = behav_data.dropna(axis=0,subset=['pairs_id'])
            start_time = behav_data['fixation.started_raw'].min() - 1 
            
            # M1 event
            onset = behav_data['pic1_render.started_raw'] - start_time
            duration = behav_data['pic2_render.started_raw'] - behav_data['pic1_render.started_raw']
            angle = behav_data['angles']
            tsv_data_m1 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})        
            tsv_data_m1['trial_type'] = 'M1'
            tsv_data_m1['modulation'] = 1
        
            # M2 event and hexagon modulation
            onset = behav_data['pic2_render.started_raw'] - start_time
            duration = behav_data['cue1.started_raw'] - behav_data['pic2_render.started_raw']
            angle = behav_data['angles']
            tsv_data_m2 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_m2['trial_type'] = 'M2'
            
            tsv_data_sin = tsv_data_m2.copy()
            tsv_data_cos = tsv_data_m2.copy()
            tsv_data_sin['trial_type'] = 'sin'
            tsv_data_cos['trial_type'] = 'cos'
        
            tsv_data_m2['modulation'] = 1
            tsv_data_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
            tsv_data_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
            
            # decision 
            onset = behav_data['cue1.started_raw'] - start_time
            duration = behav_data['cue1_2.started_raw'] - behav_data['cue1.started_raw']
            angle = behav_data['angles']
            tsv_data_rule = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_rule['trial_type'] = 'decision'
            tsv_data_rule['modulation'] = 1
            
            
            tsv_data = pd.concat([tsv_data_m1,tsv_data_m2,tsv_data_sin,tsv_data_cos,tsv_data_rule],axis=0)
            tsv_save_path = join(save_dir,'sub-{}_task-game1_run-{}_events.tsv'.format(subj,run_id))
            tsv_data.to_csv(tsv_save_path, sep="\t", index=False)
            
 
                 
def testFai_effect_event(subj,runs,ifold_fai):        
    # subj:string, subject id
    # 
    # ifold_fai: npy file, save the dictionary about ifold:mean_orientation
    event_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events/sub-{}'.format(subj)    
    read_events_name = 'hexonM2Long/{}fold/sub-{}_task-game1_run-{}_events.tsv'
    
    for ifold in range(4,9):
        for run_id in runs:
            save_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events/sub-{}/fai_effect'.format(subj)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            event_path = join(event_dir,read_events_name.format(ifold,subj,run_id))
            event_test = pd.read_csv(event_path,sep='\t')
            # generate test design matrixs
            fai_modulation = event_test.loc[event_test.trial_type=='cos']
            event_test = event_test.drop(event_test.loc[event_test.trial_type=='sin'].index)
            event_test = event_test.drop(event_test.loc[event_test.trial_type=='cos'].index)
            
            angle = fai_modulation['angle']
            fai_modulation['trial_type'] = 'fai_modulation'
        
            mean_orientation = ifold_fai[ifold]
            fai_modulation['modulation'] = np.cos(np.deg2rad(ifold*(angle - mean_orientation)))
            event_test = pd.concat([event_test,fai_modulation],axis=0)
            
            save_dir = join(save_dir, '{}fold'.format(ifold))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
            event_name= 'sub-{}_task-game1_run-{}_events_fai_effect.tsv'.format(subj, run_id)
            print(ifold,'fold',event_name)
            event_test.to_csv(join(save_dir,event_name), sep='\t', index=False)
            


def hexonRule(subj):
    # hexagon modulation effect when the second monster show
    """generate the event file of hexagon modulation"""
    runs = range(1,7)
    for ifold in range(4,9):
        for idx in runs:
            run_id = str(idx)
            # make directroy
            behav_data_path = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'.format(subj,subj,run_id)
            save_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events/sub-{}/hexonRule/{}fold'.format(subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)            

            # set start point
            behav_data = pd.read_csv(behav_data_path)
            behav_data = behav_data.dropna(axis=0, subset=['pairs_id'])
            start_time = behav_data['fix_start_cue.started'][1]
            
            # M1 event
            onset = behav_data['pic1_render.started'] - start_time
            duration = behav_data['pic2_render.started'] - behav_data['pic1_render.started']
            angle = behav_data['angles']
            tsv_data_m1 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})        
            tsv_data_m1['trial_type'] = 'M1'
            tsv_data_m1['modulation'] = 1
        
            # M2 event and hexagon modulation
            onset = behav_data['pic2_render.started'] - start_time
            #duration = behav_data['cue1.started'] - behav_data['pic2_render.started']
            duration = [2.5] * len(onset)
            angle = behav_data['angles']
            tsv_data_m2 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_m2['trial_type'] = 'M2'
            tsv_data_m2['modulation'] = 1
        
            
            # decision 
            onset =  behav_data['cue1.started'] - start_time
            duration = behav_data['cue1_2.started'] - behav_data['cue1.started']
            angle = behav_data['angles']
            tsv_data_rule = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_rule['trial_type'] = 'decision'

            
            tsv_data_sin = tsv_data_rule.copy()
            tsv_data_cos = tsv_data_rule.copy()
            tsv_data_sin['trial_type'] = 'sin'
            tsv_data_cos['trial_type'] = 'cos'
        
            tsv_data_rule['modulation'] = 1       
            tsv_data_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
            tsv_data_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
            
            tsv_data = pd.concat([tsv_data_m1,tsv_data_m2,tsv_data_sin,tsv_data_cos,tsv_data_rule],axis=0)
            tsv_save_path = join(save_dir,'sub-{}_task-game1_run-{}_events.tsv'.format(subj,run_id))
            tsv_data.to_csv(tsv_save_path, sep="\t", index=False)

            
def hexonRule_v2(subj):
    # hexagon modulation effect when the second monster show
    """generate the event file of hexagon modulation"""
    runs = range(1,7)
    for ifold in range(4,9):
        for idx in runs:
            run_id = str(idx)
            
            # make directroy
            behav_data_path = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'.format(subj,subj,run_id)
            save_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events/sub-{}/hexonRule/{}fold'.format(subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)            

            # set start point
            behav_data = pd.read_csv(behav_data_path)
            behav_data = behav_data.dropna(axis=0,subset=['pairs_id'])
            start_time = behav_data['fixation.started_raw'].min() - 1 
            
            # M1 event
            onset = behav_data['pic1_render.started_raw'] - start_time
            duration = behav_data['pic2_render.started_raw'] - behav_data['pic1_render.started_raw']
            angle = behav_data['angles']
            tsv_data_m1 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})        
            tsv_data_m1['trial_type'] = 'M1'
            tsv_data_m1['modulation'] = 1
        
            # M2 event and hexagon modulation
            onset = behav_data['pic2_render.started_raw'] - start_time
            #duration = behav_data['cue1.started_raw'] - behav_data['pic2_render.started_raw']
            duration = [2.5] * len(onset)
            angle = behav_data['angles']
            tsv_data_m2 = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_m2['trial_type'] = 'M2'
            tsv_data_m2['modulation'] = 1
            
            # decision 
            onset = behav_data['cue1.started_raw'] - start_time
            duration = behav_data['cue1_2.started_raw'] - behav_data['cue1.started_raw']
            angle = behav_data['angles']
            tsv_data_rule = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            tsv_data_rule['trial_type'] = 'decision'

            tsv_data_sin = tsv_data_rule.copy()
            tsv_data_cos = tsv_data_rule.copy()
            
            tsv_data_sin['trial_type'] = 'sin'
            tsv_data_cos['trial_type'] = 'cos'

            tsv_data_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
            tsv_data_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
            tsv_data_rule['modulation'] = 1
            
            
            tsv_data = pd.concat([tsv_data_m1,tsv_data_m2,tsv_data_sin,tsv_data_cos,tsv_data_rule],axis=0)
            tsv_save_path = join(save_dir,'sub-{}_task-game1_run-{}_events.tsv'.format(subj,run_id))
            tsv_data.to_csv(tsv_save_path, sep="\t", index=False)
            
if __name__ == "__main__":
    #subjects = [6,7,8,9,10,11,12,14,15,16,24,29,32,36,43,46]
    #subjects = [17,18,21,22]
    subjects = [27]
    #subjects = [41]
    for sub in subjects:
        sub = str(sub).zfill(3)
        print('----sub-{}----'.format(sub))
        hexonM2(sub)
        hexonRule(sub)
        #hexonM2_v2(sub)
        #hexonRule_v2(sub)