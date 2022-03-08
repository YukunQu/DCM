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
    columns = behavior_data.columns
    if 'cue1.started' in columns:
        oneDtaskColumn = ['姓名','date','expName','pairs_id','pic1','pic2',
                          'ap_diff','dp_diff','correctAns',
                          'resp.keys','resp.rt','fightRule','cue1.started']
        oneDtask = behavior_data[oneDtaskColumn].dropna(axis=0,subset=['cue1.started'])
    elif 'cue1.started_raw' in columns:
        oneDtaskColumn = ['pairs_id','pic1','pic2',
                          'ap_diff','dp_diff','correctAns',
                          'resp.keys_raw','resp.rt_raw','fightRule','cue1.started_raw']
        oneDtask = behavior_data[oneDtaskColumn].dropna(axis=0,subset=['cue1.started_raw'])
        oneDtask = oneDtask.fillna('None')
    else:
        print('Please check data columns')
        
    if trial_check:
        if len(oneDtask) != 240:
            print("Trial number of the subject is not 240! You may need check data!")
    
    total_right_num = 0
    ap_right_num = 0
    dp_right_num = 0
    ap_trial_num = 0
    dp_trial_num = 0
    for index, row in oneDtask.iterrows():  
        if 'cue1.started' in columns:
            resp_trial = row['resp.keys']
        elif 'cue1.started_raw' in columns:
            resp_trial = row['resp.keys_raw']

            if resp_trial == 'None':
                resp_trial = 'None'
            else:
                resp_trial = resp_trial[1]
                
        corrAns_trial = row['correctAns']
        fightRule_trial = row['fightRule']

        if (resp_trial != 'None'):
            if float(resp_trial) == float(corrAns_trial):
                total_right_num += 1        
                if fightRule_trial == 'AP':
                    ap_right_num += 1
                elif fightRule_trial == 'DP':
                    dp_right_num += 1 
                else:
                    print('Error')

        if fightRule_trial == 'AP':
            ap_trial_num += 1
        elif fightRule_trial == 'DP':
            dp_trial_num += 1 
        else:
            print('Error')
    total_acc = round(total_right_num/len(oneDtask),3)
    ap_acc = round(ap_right_num/ap_trial_num,3)
    dp_acc = round(dp_right_num/dp_trial_num,3)
    
    return total_acc, ap_acc, dp_acc,len(oneDtask)



#%%

def game2_acc(game2_data):
    # select key columns
    columns = game2_data.columns
    if 'cue1.started' in columns:
        game2Column = ['pairs_id','pic1_ap','pic1_dp','pic2_ap','pic2_dp',
                       'dResp.keys','dResp.rt','fightRule','cue1.started']
        game2Data = game2_data[game2Column].dropna(axis=0,subset=['cue1.started'])
    elif 'cue1.started_raw' in columns:
        game2Column = ['pairs_id','pic1_ap','pic1_dp','pic2_ap','pic2_dp',
                        'dResp.keys_raw','dResp.rt_raw','fightRule','cue1.started_raw']
        game2Data = game2_data[game2Column].dropna(axis=0,subset=['cue1.started_raw'])
        game2Data = game2Data.fillna('None')
    else:
        print('Please check data columns')
        
    print('The trial number of this game2 task is ',len(game2Data))

    # calculate the correct rate
    columns = game2Data.columns
    game2Data = game2Data.fillna('None')
    if 'dResp.keys' in columns:
        keyResp_list = game2Data['dResp.keys']
    elif 'dResp.keys_raw' in columns:   
        keyResp_tmp =game2Data['dResp.keys_raw']
        keyResp_list = []
        for k in keyResp_tmp:
            if k == 'None':
                keyResp_list.append(k)
            else:
                keyResp_list.append(k[1])
                
    trial_corr = []
    for keyResp,row in zip(keyResp_list, game2Data.itertuples()):
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
    accuracy = np.round(np.sum(trial_corr) / len(game2Data),3)
    return accuracy


