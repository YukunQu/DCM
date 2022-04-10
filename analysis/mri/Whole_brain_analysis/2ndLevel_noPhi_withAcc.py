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
spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('usable==1')
pid = data['Participant_ID'].to_list()
sub_all = [p.split('_')[-1] for p in pid]

game1_acc = data['game1_acc'].to_list()

subj_config = {'adult':['010','018','023','024','027','029','033','037','043',
                        '046','047','053','056','058','062','067','068','069'],
               'adolescent':['022', '031', '032', '036', '049', '050',  # del 016
                             '055', '059','060', '061','065'],
               'children':['011', '012', '015', '017', '025', '048', '063','064'],
               'hp':['010','024','043','046','053','062','067','068','069'],
               'all':sub_all}

sub_type = 'all'
subject_list  = subj_config[sub_type]
#%%
# data input and ouput
contrast_list = ['ZF_0004']
infosource = Node(IdentityInterface(fields=['contrast_id','subj_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list)]
infosource.inputs.subj_id = subject_list

# SelectFiles - to grab the data (alternativ to DataGrabber)
data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
templates = {'cons': pjoin(data_root, 'hexagon/specificTo6/training_set/trainsetall/6fold',
                           'sub-{subj_id}','{contrast_id}.nii')}  # look out

# Create SelectFiles node
selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                   name='selectfiles', iterfield=['subj_id'])

# Initiate DataSink node here
container_path = 'hexagon/specificTo6/training_set/trainsetall/group/{}/Acc'.format(sub_type)  # look out
datasink = Node(DataSink(base_directory=data_root,
                         container=container_path),
                name="datasink")

# Use the following substitutions for the DataSink output
substitutions = [('_cont_id_', 'con_')]
datasink.inputs.substitutions = substitutions

# Node initialize
onesamplettestdes = Node(OneSampleTTestDesign(), name="onesampttestdes")
onesamplettestdes.inputs.covariates = [dict(vector=game1_acc, name='game1_acc', centering=1)]


level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

level2conestimate = Node(EstimateContrast(group_contrast=True),
                         name="level2conestimate")
# specify contrast
condition_names = ['mean','game1_acc']
cont01 = ['Group',   'T', condition_names, [1,0]]
cont02 = ['Acc',     'T', condition_names, [0,1]]
level2conestimate.inputs.contrasts = [cont01, cont02]

# 2nd workflow
# look out
analysis2nd = Workflow(name='work_2nd',
                       base_dir='/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/hexagon/{}/Acc'.format(sub_type))
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