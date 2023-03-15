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


def level2nd_onesample_ttest(subject_list, contrast_1st, configs):
    # set spm
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # read paramters
    data_root = configs['data_root']
    task = configs['task']
    glm_type = configs['glm_type']
    set_id = configs['set_id']
    ifold = configs['ifold']

    # data input and ouput
    infosource = Node(IdentityInterface(fields=['contrast_id', 'subj_id']), name="infosource")
    infosource.iterables = [('contrast_id', contrast_1st)]
    infosource.inputs.subj_id = subject_list

    # Create SelectFiles node
    templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}','sub-{subj_id}', '{contrast_id}.nii')}
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/{set_id}/{ifold}/group/mean'
    datasink = Node(DataSink(base_directory=data_root,container=container_path),
                    name="datasink")

    # Use the following substitutions for the DataSink output
    substitutions = [('_cont_id_', 'con_')]
    datasink.inputs.substitutions = substitutions

    # Node initialize
    onesamplettestdes = Node(OneSampleTTestDesign(), name="onesampttestdes")

    level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),name="level2estimate")

    level2conestimate = Node(EstimateContrast(group_contrast=True),name="level2conestimate")
    # specify contrast
    cont01 = ['Group', 'T', ['mean'], [1]]
    level2conestimate.inputs.contrasts = [cont01]

    level2thresh = Node(Threshold(contrast_index=1,
                                  use_topo_fdr=True,
                                  use_fwe_correction=False,
                                  extent_threshold=0,
                                  height_threshold=0.01,
                                  height_threshold_type='p-value',
                                  extent_fdr_p_threshold=0.05,
                                  ),
                        name="level2thresh")

    # 2nd workflow
    analysis2nd = Workflow(name='work_2nd',base_dir=os.path.join(data_root,'working_dir',container_path))
    analysis2nd.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),('subj_id', 'subj_id')]),

                         (selectfiles, onesamplettestdes, [('cons', 'in_files')]),

                         (onesamplettestdes, level2estimate, [('spm_mat_file','spm_mat_file')]),

                         (level2estimate, level2conestimate, [('spm_mat_file','spm_mat_file'),
                                                              ('beta_images','beta_images'),
                                                              ('residual_image','residual_image')]),

                         (level2conestimate, level2thresh, [('spm_mat_file','spm_mat_file'),
                                                            ('spmT_images','stat_image'),]),

                         (level2conestimate, datasink, [('spm_mat_file','2ndLevel.@spm_mat'),
                                                        ('spmT_images', '2ndLevel.@T'),
                                                        ('con_images',  '2ndLevel.@con')]),

                         (level2thresh, datasink,   [('thresholded_map', '2ndLevel.@threshold')])
                         ])
    # run 2nd analysis
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 30})


def level2nd_covariate(subject_list,contrast_1st,contrast_2nd,covariates,configs):
    # set spm
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # read parameters
    data_root = configs['data_root']
    task = configs['task']
    glm_type = configs['glm_type']
    set_id = configs['set_id']
    ifold = configs['ifold']
    covary_type = configs['covary_type']

    # data input and ouput
    infosource = Node(IdentityInterface(fields=['contrast_id', 'subj_id']), name="infosource")
    infosource.iterables = [('contrast_id', contrast_1st)]
    infosource.inputs.subj_id = subject_list

    # SelectFiles
    templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}','sub-{subj_id}', '{contrast_id}.nii')}

    # Create SelectFiles node
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/{set_id}/{ifold}/group/{covary_type}'
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

    level2thresh = Node(Threshold(contrast_index=1,
                                  use_topo_fdr=True,
                                  use_fwe_correction=False,
                                  extent_threshold=0,
                                  height_threshold=0.01,
                                  height_threshold_type='p-value',
                                  extent_fdr_p_threshold=0.05,
                                  ),
                        name="level2thresh")
    # 2nd workflow
    analysis2nd = Workflow(name='work_2nd',base_dir='{}/working_dir/'
                                                    '{}/{}/group/{}'.format(data_root,task,glm_type,covary_type))
    analysis2nd.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                                    ('subj_id', 'subj_id')]),
                         (selectfiles, onesamplettestdes, [('cons', 'in_files')]),

                         (onesamplettestdes, level2estimate, [('spm_mat_file','spm_mat_file')]),

                         (level2estimate, level2conestimate, [('spm_mat_file','spm_mat_file'),
                                                              ('beta_images','beta_images'),
                                                              ('residual_image','residual_image')]),

                         #(level2conestimate, level2thresh, [('spm_mat_file','spm_mat_file'),
                         #                                   ('spmT_images','stat_image'),]),

                         (level2conestimate, datasink, [('spm_mat_file', '2ndLevel.@spm_mat'),
                                                        ('spmT_images',  '2ndLevel.@T'),
                                                        ('con_images',   '2ndLevel.@con')]),
                         #(level2thresh, datasink,   [('thresholded_map', '2ndLevel.@threshold')])
                         ])
    # run 2nd analysis
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 30})


def level2nd_covar_acc(participants_info,contrast_1st,configs):
    pid = participants_info['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    condition_names = ['mean', 'acc']
    cont01 = ['Group',  'T', condition_names, [1, 0]]
    cont02 = ['acc',    'T', condition_names, [0, 1]]
    contrast_2nd = [cont01, cont02]

    # covariates
    task = configs['task']
    covariates = {}
    if task == 'game1':
        covariates['acc'] = participants_info['game1_acc'].to_list()
    elif task == 'game2':
        covariates['acc'] = participants_info['game2_test_acc'].to_list()
    else:
        raise Exception("Task type is wrong.")
    configs['covary_type'] = 'acc'
    level2nd_covariate(subject_list,contrast_1st,contrast_2nd,covariates,configs)


def level2nd_covar_age(participants_info,contrast_1st,configs):
    pid = participants_info['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]

    condition_names = ['mean', 'age']
    cont01 = ['Group',          'T', condition_names, [1, 0]]
    cont02 = ['age',            'T', condition_names, [0, 1]]
    contrast_2nd = [cont01, cont02]

    # covariates
    covariates = {'age': participants_info['Age'].to_list()}
    configs['covary_type'] = 'age'
    level2nd_covariate(subject_list,contrast_1st,contrast_2nd,covariates,configs)