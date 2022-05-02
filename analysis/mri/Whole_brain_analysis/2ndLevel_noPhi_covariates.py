#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:07:25 2022

@author: dell
"""

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


def level2nd_noPhi_covariate(task,glm_type,subject_list, contrast_list, covariates):
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # data input and ouput
    infosource = Node(IdentityInterface(fields=['contrast_id', 'subj_id']),
                      name="infosource")
    infosource.iterables = [('contrast_id', contrast_list)]
    infosource.inputs.subj_id = subject_list

    # SelectFiles - to grab the data (alternativ to DataGrabber)
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
    templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/Setall/6fold',
                         'sub-{subj_id}', '{contrast_id}.nii')}  # look out

    # Create SelectFiles node
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/Setall/group/covariates'
    datasink = Node(DataSink(base_directory=data_root,
                             container=container_path),
                    name="datasink")

    # Use the following substitutions for the DataSink output
    substitutions = [('_cont_id_', 'con_')]
    datasink.inputs.substitutions = substitutions

    # Node initialize
    onesamplettestdes = Node(OneSampleTTestDesign(), name="onesampttestdes")

    age = covariates['age']
    acc = covariates['acc']
    onesamplettestdes.inputs.covariates = [dict(vector=age, name='Age', centering=1),
                                           dict(vector=acc, name='Acc', centering=1)]

    level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level2estimate")

    level2conestimate = Node(EstimateContrast(group_contrast=True),
                             name="level2conestimate")
    # specify contrast
    condition_names = ['mean', 'Age', 'Acc']
    cont01 = ['Group', 'T', condition_names, [1, 0, 0]]
    cont02 = ['Age',   'T', condition_names, [0, 1, 0]]
    cont03 = ['Acc',   'T', condition_names, [0, 0, 1]]
    level2conestimate.inputs.contrasts = [cont01, cont02, cont03]

    # 2nd workflow
    # look out
    analysis2nd = Workflow(name='work_2nd',
                           base_dir='/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/'
                                    '{}/{}/group/covariates'.format(task,glm_type))
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
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 10})


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game2_fmri==1')  # look out
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('_')[-1] for p in pid]

    task = 'game2'
    glm_type = 'hexonM2short'
    contrast_list = ['ZF_0004']
    # covariates
    covariates = {}
    covariates['age'] = data['Age'].to_list()
    covariates['acc'] = data['game2_test_acc'].to_list()  # game2_test_acc

    level2nd_noPhi_covariate(task,glm_type,subject_list, contrast_list, covariates)