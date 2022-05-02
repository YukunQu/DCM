#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:52:46 2021

@author: dell
"""


import os
import pandas as pd
from analysis.behaviour.utils import game2_acc

"""calculate accuracy of game2-test task """
# subject list
data_dir = r'/mnt/workdir/DCM/sourcedata'
subjects = ['sub_'+str(i).zfill(3) for i in range(74,79)]
game2test_acc = pd.DataFrame(columns=['Participant_ID', 'game2_test_acc'])
for sub in subjects:  # sub_id-1 ~ sub_id
    if sub in ['sub_052','sub_054','sub_066']:
        sub_game2_acc = {'Participant_ID':sub,'game2_test_acc':float('nan')}
        game2test_acc = game2test_acc.append(sub_game2_acc,ignore_index=True)
        print(sub,"have",0,"runs file.")
        continue

    game2_data_dir = os.path.join(data_dir,sub,'Behaviour','fmri_task-game2-test')
    game2_file_list = []
    for i in range(1,3):
        file_name = '{}_task-game2_run-{}.csv'.format(sub.replace('_','-'),i)
        game2_file_list.append(file_name)
    game2_file_num = len(game2_file_list)
    if game2_file_num == 1:
        game2_data_path = os.path.join(game2_data_dir,game2_file_list[0])
        game2_data = pd.read_csv(game2_data_path)
    else:
        print(sub,"have",game2_file_num,"runs file.")
        game2_data = []
        for game2file in game2_file_list:
            game2_data_path = os.path.join(game2_data_dir,game2file)
            game2_data_part = pd.read_csv(game2_data_path)
            game2_data.append(game2_data_part)
        game2_data = pd.concat(game2_data)
    accuracy = game2_acc(game2_data)
    sub_game2_acc = {'Participant_ID':sub,'game2_test_acc':accuracy}
    game2test_acc = game2test_acc.append(sub_game2_acc,ignore_index=True)
    print(sub,": calculation is complete!")

#%%
participants_tsv =  r'/mnt/workdir/DCM/docs/被试招募及训练/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
for index,row in game2test_acc.iterrows():
    sub_id = row['Participant_ID']
    acc = row['game2_test_acc']
    participants_data.loc[participants_data['Participant_ID']==sub_id,'game2_test_acc'] = acc
participants_data.to_csv(participants_tsv,sep='\t',index=False)