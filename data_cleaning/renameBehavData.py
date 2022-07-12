#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:59:29 2021

@author: dell
"""

import os 
import pandas as pd 


def rename_game1_behav_data(subjects):
    for subid in subjects:
        subid = str(subid).zfill(3)
        game1_behav_data_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1'
        game1_behav_data_dir = game1_behav_data_dir.format(subid)
        file_list = os.listdir(game1_behav_data_dir)
        file_list.sort()
        print('File number of sub_{}:'.format(subid),len(file_list))
        run_num = 0
        for file in file_list:
            if ('sub' not in file) and ('.csv' in file):
                file_path = os.path.join(game1_behav_data_dir, file)
                file_data = pd.read_csv(file_path)
                if 'loop' in file:
                    run = file_data.at[45,'pic1']
                    file_data = file_data[0:42]
                else:
                    run = file_data['run'][0]
                print(run)
                file_new_name = 'sub-{}_task-game1_run-{}.csv'.format(subid,run)
                run_num += 1
                save_path = os.path.join(game1_behav_data_dir, file_new_name)
                file_data.to_csv(save_path)
        print(subid,"have",run_num,'runs file.')


def rename_game2_behav_data(subjects):
    for subid in subjects:
        subid = str(subid).zfill(3)
        game1_behav_data_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test'
        game1_behav_data_dir = game1_behav_data_dir.format(subid)
        file_list = os.listdir(game1_behav_data_dir)
        file_list.sort()
        #print('File number of sub_{}:'.format(subid),len(file_list))
        run_num = 0
        for file in file_list:
            if ('sub' not in file) and ('.csv' in file):
                file_path = os.path.join(game1_behav_data_dir, file)
                file_data = pd.read_csv(file_path)
                if 'testBlock' in file:
                    run = file_data.at[45,'pic1']
                    file_data = file_data[0:42]
                else:
                    run = file_data['run'][0]
                file_new_name = 'sub-{}_task-game2_run-{}.csv'.format(subid,run)
                run_num += 1
                save_path = os.path.join(game1_behav_data_dir, file_new_name)
                file_data.to_csv(save_path)
        print(subid,"have",run_num,'runs file.')


if __name__ =="__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    #subject_list = [p.split('_')[-1] for p in pid]
    subject_list = [str(i).zfill(3) for i in range(89,90)]
    rename_game1_behav_data(subject_list)
    