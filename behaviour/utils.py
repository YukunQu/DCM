# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:32:27 2021

@author: QYK
"""
import os
import math
import numpy as np
from os.path import join
import pandas as pd


#%%
def recall_test_acc(behavior_data):
    testsColumn = ['姓名','date','expName','pairs_id','pic1','pic2',
                      'ap_diff','dp_diff','correctAns',
                      'resp2.keys','resp2.rt','test_loop.thisRepN']
    test = behavior_data[testsColumn].dropna(axis=0,subset=['test_loop.thisRepN'])
    resp = test['resp2.keys']
    correctAns = test['correctAns']
    correct_num = 0
    for r,ans in zip(resp,correctAns):
        if r != 'None':
            if float(r) == float(ans):
                correct_num += 1
    accuracy = correct_num/len(resp)  
    return accuracy


#%%
def recall_1D_acc(behavior_data):
    # select key columns
    oneDtaskColumn = ['姓名','date','expName','pairs_id','pic1','pic2',
                      'ap_diff','dp_diff','correctAns',
                      'resp_infer.keys','resp_infer.rt','fightRule']
    oneDtask = behavior_data[oneDtaskColumn].dropna(axis=0,subset=['fightRule'])
    resp = resp = oneDtask['resp_infer.keys']
    
    if oneDtask['fightRule'].iloc[0] == 'AP':
        rank_diff = oneDtask['ap_diff']
    elif oneDtask['fightRule'].iloc[0] == 'DP':
        rank_diff = oneDtask['dp_diff']
    else:
        print('The Dimension not support.')
    
    correct_number = 0
    for r,diff in zip(resp,rank_diff):
        if ((diff > 0) & (r == 2)) | ((diff < 0) & (r == 1)):
            correct_number = correct_number + 1
            
        elif ((diff > 0) & (r == 1)) | ((diff < 0) & (r == 2)):
            correct_number = correct_number + 0
        else:
            print('Error:',r,diff)
    acc = correct_number/(len(resp))
    return acc


#%%
def meg_1D_acc(behavior_data,trial_check=True):
    # select key columns
    oneDtaskColumn = ['姓名','date','expName','pairs_id','pic1','pic2',
                      'ap_diff','dp_diff','correctAns',
                      'resp.keys','resp.rt','fightRule','cue1.started']
    oneDtask = behavior_data[oneDtaskColumn].dropna(axis=0,subset=['cue1.started'])
    
    if trial_check:
        if len(oneDtask) != 240:
            print("The trial number of subject {} is not 240! You may need check data!")
    
    total_right_num = 0
    ap_right_num = 0
    dp_right_num = 0
    ap_trial_num = 0
    dp_trial_num = 0
    for index, row in oneDtask.iterrows():    
        resp_trial = row['resp.keys']
        corrAns_trial = row['correctAns']
        fightRule_trial = row['fightRule']
        
        if resp_trial != 'None':
            if int(resp_trial) == int(corrAns_trial):
                total_right_num += 1        
                if fightRule_trial == 'AP':
                    ap_right_num += 1
                elif fightRule_trial == 'DP':
                    dp_right_num += 1 

        if fightRule_trial == 'AP':
            ap_trial_num += 1
        elif fightRule_trial == 'DP':
            dp_trial_num += 1 
    total_acc = round(total_right_num/len(oneDtask),3)
    ap_acc = round(ap_right_num/ap_trial_num,3)
    dp_acc = round(dp_right_num/dp_trial_num,3)
    
    return total_acc, ap_acc, dp_acc


#%%
def game1_acc(subject,trial_check=True):
    """Calculat the accuracy of task game1"""
    behDataDir = r'/mnt/data/Project/DCM/BIDS/sourcedata'
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
    subjects = ['019','020']
    subject_acc = {}
    for subject in subjects:
        trial_corr,accuracy = game1_acc(subject, trial_check=True)
        subject_acc[subject] = accuracy
    
