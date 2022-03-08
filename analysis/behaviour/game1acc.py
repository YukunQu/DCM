#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:51:34 2022

@author: dell
"""
import os
import math
import numpy as np
from os.path import join
import pandas as pd


def game1_acc(subject,trial_check=True):
    """Calculat the accuracy of task game1"""
    behDataDir = r'/mnt/data/Project/DCM/sourcedata'
    taskDataDir = join(behDataDir,'sub_{}'.format(subject),'Behaviour','fmri_task-game1')
    behData = []
    for i in range(1,7):
        file_name = 'sub-{}_task-game1_run-{}.xlsx'.format(subject,i)
        file_path = join(taskDataDir,file_name)
        if os.path.exists(file_path):    
            run_df = pd.read_excel(file_path)
        else:
            file_name = 'sub-{}_task-game1_run-{}.csv'.format(subject,i)
            file_path = join(taskDataDir,file_name)
            if os.path.exists(file_path):  
                run_df = pd.read_csv(file_path)
            else:
                print('Warning: The run {} did not find in the sub-{} directory'.format(i,subject))
                continue
        behData.append(run_df)
    behData = pd.concat(behData,axis=0)
    # clean and check the trial number is right
    behData = behData.dropna(subset=['pic1'])
    if trial_check:
        if len(behData) != 252:
            print('The trial number of sub-{} is not right! It is {} Please check data.'.format(subject,len(behData)))
        else:
            print('The trial number of sub-{} is 252'.format(subject))
    
    # calculate the correct rate
    columns = behData.columns
    behData = behData.fillna('None')
    if 'resp.keys' in columns:
        keyResp_list = behData['resp.keys']
    elif 'resp.keys_raw' in columns:   
        keyResp_tmp = behData['resp.keys_raw']
        keyResp_list = []
        for k in keyResp_tmp:
            if k == 'None':
                keyResp_list.append(k)
            else:
                keyResp_list.append(k[1])
                
    trial_corr = []
    for keyResp,row in zip(keyResp_list, behData.itertuples()):
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
    accuracy = np.round(np.sum(trial_corr) / len(behData),3)
    return trial_corr, accuracy


if __name__ == "__main__":
    subjects = range(65,66)
    subjects = [(str(s).zfill(3)) for s in subjects]
    #subjects.remove('054')
    #subjects.remove('041')
    subject_acc = {}
    for subject in subjects:
        trial_corr,accuracy = game1_acc(subject, trial_check=True)
        subject_acc['sub_{}'.format(subject)] = accuracy
#%%
    participants_tsv = r'/mnt/data/Project/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    for sub_id, acc in subject_acc.items():
       participants_data.loc[participants_data['Participant_ID']==sub_id,'game1_acc'] = acc
    participants_data.to_csv(participants_tsv,sep='\t',index=False)