# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:53:22 2021

@author: -
"""

import pandas as pd


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
    ap_corr_num = 0
    dp_corr_num = 0
    for r,diff in zip(resp,rank_diff):
        if ((diff > 0) & (r == 2)) | ((diff < 0) & (r == 1)):
            correct_number = correct_number + 1
            
        elif ((diff > 0) & (r == 1)) | ((diff < 0) & (r == 2)):
            correct_number = correct_number + 0
        else:
            print('Error:',r,diff)
    acc = correct_number/(len(resp))
    return acc


#
# extract data of task task

# test_accuracy = cal_accuracy(resp, correctAns)


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
    
behavior_data = pd.read_csv(r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/data/sub_012/meg_task-1DInfer/2_meg_2021_Nov_28_1036.csv')
total_acc, ap_acc, dp_acc = meg_1D_acc(behavior_data)