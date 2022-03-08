#!/usr/bin/env python
# coding: utf-8

# ## 1st-level Analysis Workflow Structure
# 
#     1. Specify 1st-level model parameters
#     2. Specify 1st-level contrasts
#     3. Estimate 1st-level contrasts
#     4. Normalize 1st-level contrasts

import time 
import os
import pandas as pd 
from os.path import join as pjoin

from nipype import Node, MapNode, Workflow
from nipype.interfaces.utility import Function,IdentityInterface

from nipype.interfaces.base import Bunch
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm import Smooth
from nipype.interfaces.spm import Level1Design
from nipype.interfaces.spm import EstimateModel
from nipype.interfaces.spm import EstimateContrast

from nipype import SelectFiles
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import DataSink
from nipype.interfaces import spm

# Specify which SPM to use
starttime = time.time()
spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

def run_info(ev_file,motions_file=None):
    import pandas as pd 
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations  = []

    pmod_names  = []
    pmod_params = []
    pmod_polys  = []
    
    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2_corr','M2_error','decision']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    motions_df = pd.read_csv(motions_file,sep='\t')
    
    motion_columns   = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                        'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                        'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
                        'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
                        'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                        'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
    
    """motion_columns= ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']"""

    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()
            
    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,pmod=[None,run_pmod,None,None],
                     orth=['No','No','No','No'],regressor_names=motion_columns,regressors = motions)
    
    return run_info

#%%
# specify subjects
participants_tsv = r'/mnt/data/Project/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('usable==1')
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]
#subject_list = ['010','011','012','015','023','033','040','048','049','050',
#                '055','056','058','059','060','061','062','063','064']
#subject_list = ['010','024','032','036','043','046','053','061','062']
# set parameters
tr = 3.
runs_list = [1,2,3]
analysis_type = 'M2short'

# input files
data_root = '/mnt/data/Project/DCM/BIDS/derivatives/fmriprep_volume'
event_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/Events'
templates = {'func': pjoin(data_root,'sub-{subj_id}/func/sub-{subj_id}_task-game1_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
             'event': pjoin(event_dir,'sub-{subj_id}/hexonM2short/6fold/sub-{subj_id}_task-game1_run-{run_id}_events.tsv'),
             'regressors':pjoin(data_root,'sub-{subj_id}/func/sub-{subj_id}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv')
            }

# outputfiles
datasink_dir = '/mnt/data/Project/DCM/BIDS/derivatives/Nipype/'
working_dir = '/mnt/data/Project/DCM/BIDS/derivatives/Nipype/working_dir'
#%%
# Specify input & output stream
infosource = Node(IdentityInterface(fields=['subj_id']),name="infosource")
infosource.iterables = [('subj_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
selectfiles = Node(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                   name='selectfiles') 
selectfiles.inputs.run_id = runs_list

# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory=datasink_dir,
                         container=analysis_type),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_subj_id_', 'sub-')]
datasink.inputs.substitutions = substitutions

# Specify GLM contrasts
# Condition names
condition_names = ['M1','M2_corr','M2_corrxcos^1','M2_corrxsin^1','decision']

# Contrasts
cont01 = ['M2_corrxcos^1',   'T', condition_names, [0, 0, 1, 0, 0]]          
cont02 = ['M2_corrxsin^1',   'T', condition_names, [0, 0, 0, 1, 0]]         
cont03 = ['decision',        'T', condition_names, [0, 0, 0, 0, 1]]
cont05 = ['M2_corr',         'T', condition_names, [0, 1, 0, 0, 0]]

cont04 = ['hexagon_mod',    'F', [cont01, cont02]]
contrast_list = [cont01, cont02, cont03, cont04, cont05]


# Specify Nodes
gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

# prepare event file
runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                             output_names=['run_info'],
                             function=run_info),
                    name='runsinfo',
                    iterfield = ['ev_file','motions_file'])

# SpecifyModel - Generates SPM-specific Model
modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                input_units='secs',
                                output_units='secs',
                                time_repetition=tr,
                                high_pass_filter_cutoff=128.,
                                ),
                name='modelspec')

# Level1Design - Generates an SPM design matrix
level1design = Node(Level1Design(bases={'hrf': {'derivs': [0,0]}},
                                 timing_units='secs',
                                 interscan_interval=3., 
                                 model_serial_correlations='AR(1)',
                                 flags = {'mthresh':0}),
                    name="level1design")

# EstimateModel - estimate the parameters of the model
level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                      name="level1estimate")

# EstimateContrast - estimates contrasts
level1conest = Node(EstimateContrast(contrasts=contrast_list),
                    name="level1conest")

# Specify Workflow
# Initiation of the 1st-level analysis workflow
analysis1st = Workflow(name='work_1st', base_dir=working_dir)

# Connect up the 1st-level analysis components
analysis1st.connect([(infosource, selectfiles,  [('subj_id','subj_id')]),
                     (selectfiles, runs_prep,   [('event','ev_file'),
                                                 ('regressors','motions_file')
                                                ]),
                     (runs_prep, modelspec,     [('run_info','subject_info')]),
                     (selectfiles, gunzip_func, [('func','in_file')]),
                     (gunzip_func, smooth,      [('out_file','in_files')]),
                     (smooth, modelspec,        [('smoothed_files','functional_runs')]),
                     
                     (modelspec,level1design,[('session_info','session_info')]),
                     (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),
                     
                     (level1estimate, level1conest, [('spm_mat_file','spm_mat_file'),
                                                     ('beta_images','beta_images'),
                                                     ('residual_image','residual_image')
                                                    ]),
                     (level1conest, datasink, [('spm_mat_file','1stLevel_part1.@spm_mat'),
                                               ('spmT_images', '1stLevel_part1.@T'),
                                               ('con_images',  '1stLevel_part1.@con'),
                                               ('spmF_images', '1stLevel_part1.@F'),
                                              ])
                    ])


# Visualize the workflow
# Create 1st-level analysis output graph
analysis1st.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=pjoin(analysis1st.base_dir,'work_1st', 'graph.png'))

# run the 1st analysis
analysis1st.run('MultiProc', plugin_args={'n_procs': 35})