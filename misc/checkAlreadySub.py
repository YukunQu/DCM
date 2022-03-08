#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 21:16:42 2022

@author: dell
"""
import os
import pandas as pd


participants_tsv = r'/mnt/data/Project/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')

data = participants_data.query('usable==1')
pid = data['Participant_ID'].to_list()
pid = [p.replace('_','-') for p in pid]

target_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Nipype/M2/1stLevel'
sub_list = os.listdir(target_dir)

p_notin = []
for p in pid:
    if p in sub_list:
        continue
    else:
        print(p)
        p_notin.append(p)