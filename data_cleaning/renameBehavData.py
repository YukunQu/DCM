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



if __name__ =="__main__":
    subjects= range(70,74)
    rename_game1_behav_data(subjects)
    