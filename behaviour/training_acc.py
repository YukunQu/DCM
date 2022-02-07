# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 09:27:34 2021

@author: QYK
"""
#%%
import os
import pandas as pd
from os.path import join
#%%
participant_df = pd.read_csv(r'/mnt/data/Project/DCM/participants.csv')
participant_id = participant_df['Participant_ID']

trainDir = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/result'
run1_acc = pd.read_csv(join(trainDir,'training','train_recall_run1.csv'))
run2_acc = pd.read_csv(join(trainDir,'training','train_recall_run2.csv'))

recall_acc = pd.concat([run1_acc,run2_acc])
recall_acc.sort_values("姓名",inplace=True)
recall_acc.to_csv(join(trainDir,'training','train_recall_acc.csv'))
#%%
def genSubAcc(trainDataDir,participant_id,sub_exp_id=None):
    file_tmp = os.listdir(trainDataDir)
    
    file_list = []
    acc_list = []
    time_list = []
    id_list = []
    name_list = []
    dim_list = []
    
    
    for file_name in file_tmp:
        if '.csv' in file_name:
            file_list.append(file_name)
   
    for file in file_list:
        data_path = os.path.join(trainDataDir,file)
        if os.path.getsize(data_path) > 5120:
            data_tmp = pd.read_csv(data_path)
        else:
            continue
        if data_tmp.empty:
            continue
        columns = data_tmp.columns
        if 'test_accuracy' not in columns:
            print("The participant has not done the test.")
        else:
            test_acc = data_tmp['test_accuracy'].max()
            print(test_acc)
            acc_list.append(test_acc)
            time_list.append(data_tmp['date'][0])
            id_list.append(data_tmp['participant'][0])
            name_list.append(data_tmp['姓名'][0])
            #data_tmp = data_tmp.dropna(axis=0,subset=['fightRule'])
            #dim_list.append(data_tmp['fightRule'].iloc[0])
    pids= len(name_list) * participant_id
    #sub_score = pd.DataFrame({'姓名':name_list,'Participant_ID':pids,'Exp_ID':id_list,'date':time_list,'dim':dim_list,'test_accuracy':acc_list})
    sub_score = pd.DataFrame({'姓名':name_list,'Participant_ID':pids,'Exp_ID':id_list,'date':time_list,'test_accuracy':acc_list})
    print(sub_score)     
    return sub_score
    

if __name__ == "__main__":
    participant_df = pd.read_csv(r'/mnt/data/Project/DCM/participants.csv')
    participant_id = participant_df['Participant_ID']
    Exp_id = participant_df['Exp_ID']
    
    train_acc = []
    for pid,expid in zip(participant_id,Exp_id):
        sub_train_dataDir = r'/mnt/data/Project/DCM/BIDS/sourcedata/{}/Behaviour/train_recall_run2'.format(pid)
        sub_score = genSubAcc(sub_train_dataDir,expid)
        train_acc.append(sub_score)
    train_acc = pd.concat(train_acc)
    saveDir = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/result'
    train_acc.to_csv(join(saveDir,'training','total_test.csv'))