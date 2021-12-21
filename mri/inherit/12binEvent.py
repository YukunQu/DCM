# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:14:02 2021

@author: QYK
"""
import os
from os.path import join
import pandas as pd
import numpy as np

# 12 bin 

subj = 'sub-005'
event_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\events'.format(subj) 
events_name = 'hexagon_on_M2/{}fold/{}_task-game1_run-{}_events.tsv' 
ifold_fai= np.load(r'D:\Data\Development_Cognitive_Map\bids\derivatives\nilearn\{}/ifold_fat.npy'.format(subj),allow_pickle=True).item()
run_list =  range(1,10)
ifold = 6

event_all = []
for run_id in run_list:
    event_path = join(event_dir,events_name.format(ifold,subj,run_id))
    event_test = pd.read_csv(event_path,sep='\t')
    # generate test design matrixs
    alignvsmisalign = event_test.loc[event_test.trial_type=='M2'].copy()
    event_test = event_test.drop(event_test.loc[event_test.trial_type=='sin'].index)
    event_test = event_test.drop(event_test.loc[event_test.trial_type=='cos'].index)
    
    mean_orientation = ifold_fai[ifold]
    angles = alignvsmisalign['angle']
    # use binNum label each angle 
    alignedD_360 = [(a - mean_orientation)% 360 for a in angles]    
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if a==13 else a for a in anglebinNum]
    anglebinNum = [str(binNum) for binNum in anglebinNum]
    alignvsmisalign['trial_type'] = anglebinNum
    alignvsmisalign.append(alignvsmisalign)
    
    event_test = pd.concat([event_test,alignvsmisalign],axis=0)
    event_all.append(event_test)
    
event_all = pd.concat(event_all,axis=0)
save_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\{}\events\12bin/{}fold'.format(subj,ifold)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

event_name= '{}_task-game1_run-all_events.tsv'.format(subj)
print(event_name)
event_all.to_csv(join(save_dir,event_name),sep='\t', index=False)