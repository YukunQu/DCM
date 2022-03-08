# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:53:22 2021

@author: -
"""

import os
import pandas as pd
from analysis.behaviour.utils import meg_1D_acc

"""calculate accuracy of MEG 1D task """
# subject list
beh_data_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/data'
subjects = os.listdir(beh_data_dir)
subjects.sort()
meg_1d_acc = pd.DataFrame(columns=['Participant_ID', '1D_acc', '1D_ap', '1D_dp'])
for sub in subjects[48:64]:
    if (sub == 'sub_013') or (sub =='sub_006') or (sub =='sub_035'):
        continue
    meg_data_dir = os.path.join(beh_data_dir,sub,'meg_task-1DInfer')
    meg_file_list = os.listdir(meg_data_dir)
    meg_file_num = len(meg_file_list)
    if meg_file_num == 1:
        meg_data_path = os.path.join(meg_data_dir,meg_file_list[0])
        meg_data = pd.read_csv(meg_data_path)
        Exp_id = meg_file_list[0].split('_')[0]
    else:
        print(sub,"have",meg_file_num,"runs file.")
        meg_data = []
        for mfile in meg_file_list:
            meg_data_path = os.path.join(meg_data_dir,mfile)
            meg_data_part = pd.read_csv(meg_data_path)
            meg_data.append(meg_data_part)
        meg_data = pd.concat(meg_data)
        Exp_id = mfile.split('_')[0]
    total_acc, ap_acc, dp_acc,len1dtask = meg_1D_acc(meg_data)
    
    sub_meg_acc = {'Participant_ID':sub,'1D_acc':total_acc,'1D_ap':ap_acc,'1D_dp':dp_acc}
    meg_1d_acc = meg_1d_acc.append(sub_meg_acc,ignore_index=True)
#%%
participant_tsv = r'/mnt/data/Project/DCM/BIDS/participants.tsv'
participant_data = pd.read_csv(participant_tsv,sep='\t')

for index,row in meg_1d_acc.iterrows():
    sub_id = row['Participant_ID']
    acc_1d = row['1D_acc']
    ap_1d = row['1D_ap']
    dp_1d = row['1D_dp']
    participant_data.loc[participant_data['Participant_ID']==sub_id,'1D_acc'] = acc_1d
    participant_data.loc[participant_data['Participant_ID']==sub_id,'1D_ap'] = ap_1d
    participant_data.loc[participant_data['Participant_ID']==sub_id,'1D_dp'] = dp_1d
    participant_data.to_csv(participant_tsv,sep='\t',index=False)