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


def alignPhi_2ndLevel(subject_list, contrast_list, set_id, configs):
    # Specify which SPM to use
    from nipype.interfaces import spm
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    infosource = Node(IdentityInterface(fields=['contrast_id','subj_id']),
                      name="infosource")
    infosource.iterables = [('contrast_id', contrast_list)]
    infosource.inputs.subj_id = subject_list

    # SelectFiles - to grab the data (alternativ to DataGrabber)
    data_root = configs['data_root']
    task = configs['task']
    glm_type = configs['glm_type']
    ROI = configs['ROI']
    sub_type = configs['sub_type']

    templates = {'cons': pjoin(data_root, task, glm_type,ROI,f'Set{set_id}','6fold',
                               'sub-{subj_id}','{contrast_id}.nii')}  # look out

    # Create SelectFiles node
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/{ROI}/Set{set_id}/group/6fold/{sub_type}'   # look out
    datasink = Node(DataSink(base_directory=data_root,
                             container=container_path),
                    name="datasink")

    # Use the following substitutions for the DataSink output
    substitutions = [('_cont_id_', 'con_')]
    datasink.inputs.substitutions = substitutions

    # Node initialize
    onesamplettestdes = Node(OneSampleTTestDesign(), name="onesampttestdes")

    level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level2estimate")

    level2conestimate = Node(EstimateContrast(group_contrast=True),
                             name="level2conestimate")
    # specify contrast
    cont01 = ['Group', 'T', ['mean'], [1]]
    level2conestimate.inputs.contrasts = [cont01]

    # 2nd workflow
    # look out
    analysis2nd = Workflow(name= 'work_2nd',
                           base_dir=f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir'
                                    f'/{task}/{glm_type}/{ROI}/Set{set_id}/{sub_type}')
    analysis2nd.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                                    ('subj_id','subj_id')]),
                         (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                         (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                               'spm_mat_file')]),
                         (level2estimate, level2conestimate, [('spm_mat_file',
                                                               'spm_mat_file'),
                                                              ('beta_images',
                                                               'beta_images'),
                                                              ('residual_image',
                                                               'residual_image')]),
                         (level2conestimate, datasink, [('spm_mat_file',
                                                         '2ndLevel.@spm_mat'),
                                                        ('spmT_images',
                                                         '2ndLevel.@T'),
                                                        ('con_images',
                                                         '2ndLevel.@con')])
                         ])
    # run 2nd analysis
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 10})

#%%
if __name__ == "__main__":
    # Configs files
    configs = {'data_root': '/mnt/workdir/DCM/BIDS/derivatives/Nipype',
               'task':'game1',
               'glm_type':'alignPhi',
               'ROI':'EC_func',
               'sub_type':'hp'}

    # Specify the subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query('game1_fmri==1')  # look out

    adult_data = data.query('Age>18')
    adolescent_data = data.query('12<Age<=18')
    children_data = data.query('Age<=12')
    hp_data = data.query('game1_acc>=0.80')

    # Specify the contrast list
    contrast_list = ['ZT_0001','ZT_0002','ZT_0003']

    # split 2 test set
    test_sets = [1,2]

    # ['adult','adolescent','children','hp']
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

        configs['sub_type'] = sub_type

        for set_id in test_sets:
            alignPhi_2ndLevel(subject_list,contrast_list,set_id,configs)