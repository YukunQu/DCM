#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:52:46 2021

@author: dell
"""


import os
import pandas as pd
from behaviour.utils import game2_acc

"""calculate accuracy of game2-test task """
# subject list
data_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata'
subjects = os.listdir(data_dir)
subjects.sort()
game2test_acc = pd.DataFrame(columns=['Participant_ID', 'game2_test_acc'])
for sub in subjects[27:48]:
    if sub in ['sub_013','sub_034','sub_040','sub_041','sub_035']:
        sub_game2_acc = {'Participant_ID':sub,'game2_test_acc':float('nan')}
        game2test_acc = game2test_acc.append(sub_game2_acc,ignore_index=True)
        continue

    game2_data_dir = os.path.join(data_dir,sub,'Behaviour','fmri_task-game2-test')
    game2_file_tmp = os.listdir(game2_data_dir)
    game2_file_list = []
    for f in game2_file_tmp:
        if '.csv' in f :
            game2_file_list.append(f)
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
game2test_acc_savePath = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/result/fmri_task-game2-test/game2test_acc.csv'
game2test_acc.to_csv(game2test_acc_savePath,index=False)

#participant_df = pd.read_excel(r'/mnt/data/Project/DCM/participants.xlsx')
##participant_df = pd.merge(participant_df, game2test_acc, on='Participant_ID')
#participant_df.to_csv(r'/mnt/data/Project/DCM/participants.csv',index=False)

#%%
subject_exp_path = r'/mnt/data/Project/DCM/participants_exp.xlsx'
subject_exp = pd.read_excel(subject_exp_path)
subjects_id = game2test_acc['Participant_ID']
subs_acc = game2test_acc['game2_test_acc']
for sub_id, acc in zip(subjects_id,subs_acc):
   subject_exp.loc[subject_exp['Participant_ID']==sub_id,'game2_test_acc'] =acc
subject_exp.to_excel(subject_exp_path,index=False)