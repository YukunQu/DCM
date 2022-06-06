#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:07:25 2022

@author: dell
"""

import pandas as pd
from os.path import join as pjoin

# Get the Node and Workflow object
from nipype import Node, MapNode,Workflow

from nipype.interfaces.spm import OneSampleTTestDesign
from nipype.interfaces.spm import EstimateModel, EstimateContrast
from nipype.interfaces.spm import Threshold

# Import the SelectFiles
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface

# Specify which SPM to use
from nipype.interfaces import spm
from analysis.mri.Whole_brain_analysis.secondLevel import level2nd_noPhi


if __name__ == "__main__":
    task = 'game1'
    glm_type = 'separate_hexagon'

    contrast_list = ['ZF_0005','ZF_0006','ZT_0007','ZT_0008','ZF_0011']

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query('game1_fmri==1')  # look out

    adult_data = data.query('Age>18')
    adolescent_data = data.query('12<Age<=18')
    children_data = data.query('Age<=12')
    hp_data = data.query('game1_acc>=0.8')

    print("Participants:", len(data))
    print("Adult:",len(adult_data))
    print("Adolescent:",len(adolescent_data))
    print("Children:", len(children_data))
    print("High performance:",len(hp_data),"({} adult)".format(len(hp_data.query('Age>18'))))

    set_id = 'Set1'

    for sub_type in ['hp']:
        if sub_type == 'adult':
            pid = adult_data['Participant_ID'].to_list()
        elif sub_type == 'adolescent':
            pid = adolescent_data['Participant_ID'].to_list()
        elif sub_type == 'children':
            pid = children_data['Participant_ID'].to_list()
        elif sub_type == 'hp':
            pid = hp_data['Participant_ID'].to_list()
        else:
            pid = None
        subject_list = [p.split('_')[-1] for p in pid]
        print(f"The {sub_type} group have {len(subject_list)} subjects")

        level2nd_noPhi(subject_list,sub_type,task,glm_type, set_id, contrast_list)