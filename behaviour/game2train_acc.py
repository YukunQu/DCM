#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:53:47 2021

@author: dell
"""
import os 
import numpy as np
import pandas as pd


def cal_acc(resp_dim,correctAns):
    correct_num = 0
    for r,ans in zip(resp_dim,correctAns):
        if r != 'None':
            if float(r) == float(ans):
                correct_num += 1
    accuracy = np.round(correct_num/len(resp_dim),3)
    return accuracy


def sub_game2train_acc(sub_id,game2Data):
    game2Data = game2Data.dropna(axis=0,subset=['dim'])
    
    resp_ap = []
    resp_dp = []
    corrAns_ap = []
    corrAns_dp = []
    for index, row in game2Data.iterrows():    
        dim = row['dim']
        if dim == 'ap':
            resp_ap.append(row['resp.keys'])
            corrAns_ap.append(row['correctAns'])
        elif dim == 'dp':
            resp_dp.append(row['resp.keys'])
            corrAns_dp.append(row['correctAns'])
        else:
            print("Dimension error.")
    
    print(sub_id,"have",len(resp_ap),'trials in AP dimension of game2train task.')
    ap_acc = cal_acc(resp_ap, corrAns_ap)
    print(sub_id,"have",len(resp_dp),'trials in DP dimension of game2train task.')
    dp_acc = cal_acc(resp_dp, corrAns_dp)
    
    return ap_acc,dp_acc

    
if __name__== "__main__":
    source_data_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata'
    subjects = os.listdir(source_data_dir)
    subjects.sort()
    game2_acc = pd.DataFrame(columns=['Participant_ID','game2_train_ap','game2_train_dp'])
    for sub in subjects[28:49]:
        if sub in ['sub_013','sub_034','sub_040','sub_041','sub_035']:
            continue
        sub_game2train_dir = os.path.join(source_data_dir,sub,'Behaviour','fmri_task-game2-train')
        sub_game2train_list = os.listdir(sub_game2train_dir)
        game2Data = pd.read_csv(os.path.join(sub_game2train_dir,sub_game2train_list[0]))
        ap_acc, dp_acc = sub_game2train_acc(sub, game2Data)
        sub_game2acc = {'Participant_ID':sub,'game2_train_ap':ap_acc,'game2_train_dp':dp_acc}
        game2_acc = game2_acc.append(sub_game2acc,ignore_index=True)
    game2_acc.to_csv(r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/result/fmri_task-game2-train/game2_acc.csv',index=False)