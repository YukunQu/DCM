#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:07:25 2022

@author: dell
"""
import os

import pandas as pd
from os.path import join as pjoin

# Get the Node and Workflow object
from nipype import Node, MapNode, Workflow

from nipype.interfaces.spm import OneSampleTTestDesign
from nipype.interfaces.spm import EstimateModel, EstimateContrast
from nipype.interfaces.spm import Threshold

# Import the SelectFiles
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface

# Specify which SPM to use
from nipype.interfaces import spm


def level2nd_noPhi(subject_list,sub_type,task,glm_type,set_id,contrast_1st):
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # data input and ouput
    infosource = Node(IdentityInterface(fields=['contrast_id', 'subj_id']), name="infosource")
    infosource.iterables = [('contrast_id', contrast_1st)]
    infosource.inputs.subj_id = subject_list

    # SelectFiles
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
    templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/{set_id}/6fold','sub-{subj_id}', '{contrast_id}.nii')}

    # Create SelectFiles node
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/{set_id}/group/{sub_type}'
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

    level2thresh = Node(Threshold(contrast_index=1,extent_fdr_p_threshold=0.05),
                        name="level2thresh")

    # 2nd workflow
    analysis2nd = Workflow(name='work_2nd',base_dir='/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/'
                                                    '{}/{}/{}/group/{}'.format(task,glm_type,set_id,sub_type))
    analysis2nd.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                                    ('subj_id', 'subj_id')]),
                         (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                         (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                               'spm_mat_file')]),
                         (level2estimate, level2conestimate, [('spm_mat_file',
                                                               'spm_mat_file'),
                                                              ('beta_images',
                                                               'beta_images'),
                                                              ('residual_image',
                                                               'residual_image')]),
                         (level2conestimate, level2thresh, [('spm_mat_file',
                                                             'spm_mat_file'),
                                                            ('spmT_images',
                                                             'stat_image'),
                                                            ]),
                         (level2conestimate, datasink, [('spm_mat_file',
                                                         '2ndLevel.@spm_mat'),
                                                        ('spmT_images',
                                                         '2ndLevel.@T'),
                                                        ('con_images',
                                                         '2ndLevel.@con')]),
                         (level2thresh, datasink,   [('thresholded_map',
                                                      '2ndLevel.@threshold')])
                         ])
    # run 2nd analysis
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 30})


def level2nd_noPhi_covariate(subject_list,task,glm_type,contrast_1st,contrast_2nd,covariates,covar_type):
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # data input and ouput
    infosource = Node(IdentityInterface(fields=['contrast_id', 'subj_id']), name="infosource")
    infosource.iterables = [('contrast_id', contrast_1st)]
    infosource.inputs.subj_id = subject_list

    # SelectFiles
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
    templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/Setall/6fold','sub-{subj_id}', '{contrast_id}.nii')}

    # Create SelectFiles node
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/Setall/group/covariates/{covar_type}'
    datasink = Node(DataSink(base_directory=data_root, container=container_path),
                    name="datasink")

    # Use the following substitutions for the DataSink output
    substitutions = [('_cont_id_', 'con_')]
    datasink.inputs.substitutions = substitutions

    # Node initialize
    onesamplettestdes = Node(OneSampleTTestDesign(), name="onesampttestdes")

    spmf_covariates = []
    for key,value in covariates.items():
        spmf_covariates.append(dict(vector=value, name=key, centering=1))
    onesamplettestdes.inputs.covariates = spmf_covariates

    level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level2estimate")

    level2conestimate = Node(EstimateContrast(group_contrast=True),
                             name="level2conestimate")
    # specify contrast
    level2conestimate.inputs.contrasts = contrast_2nd

    # 2nd workflow
    analysis2nd = Workflow(name='work_2nd',base_dir='/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/'
                                                    '{}/{}/group/covariates/{}'.format(task,glm_type,covar_type))
    analysis2nd.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                                    ('subj_id', 'subj_id')]),
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
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 50})


def level2nd_covar_acc(participants_info,task,glm_type,contrast_1st):
    pid = participants_info['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    condition_names = ['mean', 'acc']
    cont01 = ['Group',  'T', condition_names, [1, 0]]
    cont02 = ['acc',    'T', condition_names, [0, 1]]
    contrast_2nd = [cont01, cont02]

    # covariates
    covariates = {}
    if task == 'game1':
        covariates['acc'] = participants_info['game1_acc'].to_list()
    elif task == 'game2':
        covariates['acc'] = participants_info['game2_test_acc'].to_list()
    else:
        raise Exception("Task type is wrong.")
    covar_dir = 'acc'
    level2nd_noPhi_covariate(subject_list,task,glm_type,contrast_1st, contrast_2nd, covariates,covar_dir)


def level2nd_covar_age(participants_info,task,glm_type,contrast_1st):
    pid = participants_info['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    condition_names = ['mean', 'age']
    cont01 = ['Group',          'T', condition_names, [1, 0]]
    cont02 = ['age',            'T', condition_names, [0, 1]]
    contrast_2nd = [cont01, cont02]

    # covariates
    covariates = {}
    covariates['age'] = participants_info['Age'].to_list()
    covar_dir = 'age'
    level2nd_noPhi_covariate(subject_list,task,glm_type,contrast_1st, contrast_2nd, covariates,covar_dir)


def level2nd_covar_acc_age(paricipants_info,task,glm_type,contrast_1st):
    pid = paricipants_info['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    condition_names = ['mean', 'acc', 'age']
    cont01 = ['Group',  'T', condition_names, [1, 0, 0]]
    cont02 = ['acc',    'T', condition_names, [0, 1, 0]]
    cont03 = ['age',    'T', condition_names, [0, 0, 1]]
    contrast_2nd = [cont01, cont02, cont03]

    # covariates
    covariates = {}
    if task == 'game1':
        covariates['acc'] = paricipants_info['game1_acc'].to_list()
    elif task == 'game2':
        covariates['acc'] = paricipants_info['game2_test_acc'].to_list()
    else:
        raise Exception("Task type is wrong.")
    covariates['age'] = paricipants_info['Age'].to_list()
    covar_dir = 'acc_age'
    level2nd_noPhi_covariate(subject_list,task,glm_type,contrast_1st, contrast_2nd, covariates,covar_dir)