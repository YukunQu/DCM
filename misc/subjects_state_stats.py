#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:48:56 2022

@author: dell
"""

import os
import pandas as pd

participants_tsv = r'/mnt/data/Project/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')

data = participants_data.query('usable==1')

adult_data = data.query('Age>18')
adolescent_data = data.query('(Age>12)and(Age<=18)')
children_data = data.query('Age<=12')
hp_sub = data.query('game1_acc>0.75')

print("Number of adults:",len(adult_data))
print("Number of adolescents:",len(adolescent_data))
print("Number of children:",len(children_data))

print('Adult data',adult_data)
print("——————————————————————————————")
print('Adolescent data',adolescent_data)
print("——————————————————————————————")
print('Children data', children_data)
print("——————————————————————————————")
print("high performance subjects",hp_sub)