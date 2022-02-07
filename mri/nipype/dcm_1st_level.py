#!/usr/bin/env python
# coding: utf-8

# ## 1st-level Analysis Workflow Structure
# 
#     1. Specify 1st-level model parameters
#     2. Specify 1st-level contrasts
#     3. Estimate 1st-level contrasts
#     4. Normalize 1st-level contrasts

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import pandas as pd 
from os.path import join as pjoin

# Get the Node and Workflow object
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


# In[2]:


def run_info(ev_file,head_motions=None,regressors_file=None):
    import pandas as pd 
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations  = []

    pmod_names  = []
    pmod_params = []
    pmod_polys  = []
    
    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2','decision']
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

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,pmod=[None,run_pmod,None])
    #run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations)
        
    motions = []
    
    regressors = []
            
    return run_info,motions,regressors


# ## Experiment parameters

# In[3]:


# Specify which SPM to use
from nipype.interfaces import spm
spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')
tr = 3.


# ## Specify GLM contrasts

# In[4]:


# Condition names
condition_names = ['M1','M2','M2xcos^1','M2xsin^1','decision']

# Contrasts
cont01 = ['M2xcos^1',       'T', condition_names, [0, 0, 1, 0, 0]]          
cont02 = ['M2xsin^1',       'T', condition_names, [0, 0, 0, 1, 0]]         
cont03 = ['decision',       'T', condition_names, [0, 0, 0, 0, 1]]

cont04 = ['hexagon_mod',    'F', [cont01, cont02]]
contrast_list = [cont01, cont02, cont03, cont04]


# ## Specify input & output stream

# In[5]:


# Infosource - a function free node to iterate over the list of subject names
subject_list = ['005']
infosource = Node(IdentityInterface(fields=['subj_id']),name="infosource")
infosource.iterables = [('subj_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
data_root = '/mnt/data/Project/DCM/BIDS/derivatives/fmriprep_surfer'
templates = {'func': pjoin(data_root,'sub-{subj_id}/func/sub-{subj_id}_task-game1_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
             'event': pjoin(data_root,'sub-{subj_id}/events/hexagon_on_M2/6fold/sub-{subj_id}_task-game1_run-{run_id}_events.tsv')}
selectfiles = Node(SelectFiles(templates,base_directory=data_root,sort_filelist=True),
                   name='selectfiles') 
selectfiles.inputs.run_id = [1,2,3,4,5,6,7,8,9] 
                           
# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory='/mnt/data/Project/DCM/BIDS/derivatives/Nipype/',
                         container='result'),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_subj_id_', 'sub-')]
datasink.inputs.substitutions = substitutions


# ## Specify Nodes

# In[6]:


gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

# prepare event file
runs_prep = MapNode(Function(input_names=['ev_file','motions_file','regressors_file'],
                             output_names=['run_info','motions','regressors'],
                             function=run_info),
                    name='runsinfo',
                    iterfield = ['ev_file'])
                    # iterfield = ['ev_file','motions_file','regressors_file'])

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
                                 model_serial_correlations='AR(1)'),
                    name="level1design")

# EstimateModel - estimate the parameters of the model
level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                      name="level1estimate")

# EstimateContrast - estimates contrasts
level1conest = Node(EstimateContrast(contrasts=contrast_list),
                    name="level1conest")


# ## Specify Workflow

# In[7]:


# Initiation of the 1st-level analysis workflow
analysis1st = Workflow(name='work_1st', base_dir='/mnt/data/Project/DCM/BIDS/derivatives/Nipype/working_dir')

# Connect up the 1st-level analysis components
analysis1st.connect([(infosource, selectfiles,  [('subj_id','subj_id')]),
                     (selectfiles, runs_prep,   [('event','ev_file')]),
                     (runs_prep, modelspec,     [('run_info','subject_info')]),
                     (selectfiles, gunzip_func, [('func','in_file')]),
                     (gunzip_func, smooth,      [('out_file','in_files')]),
                     (smooth, modelspec,        [('smoothed_files','functional_runs')]),
                     #(selectfiles, modelspec,   [('func','functional_runs')]),
                     
                     (modelspec,level1design,[('session_info','session_info')]),
                     (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),
                     
                     (level1estimate, level1conest, [('spm_mat_file','spm_mat_file'),
                                                     ('beta_images','beta_images'),
                                                     ('residual_image','residual_image')]),
                     (level1conest, datasink, [('spm_mat_file','1stLevel.@spm_mat'),
                                               ('spmT_images', '1stLevel.@T'),
                                               ('con_images',  '1stLevel.@con'),
                                               ('spmF_images', '1stLevel.@F'),
                                              ])
                    ])


# In[8]:


# Initiation of the 1st-level analysis workflow
analysis1st_test = Workflow(name='work_1st', base_dir='/mnt/data/Project/DCM/BIDS/derivatives/Nipype/working_dir')
analysis1st_test.connect([(infosource, selectfiles,[('subj_id','subj_id')]),
                     (selectfiles, runs_prep, [('event','ev_file')]),
                    ])


# ## Visualize the workflow

# In[9]:


# Create 1st-level analysis output graph
analysis1st.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=pjoin(analysis1st.base_dir,'work_1st', 'graph.png'))


# In[10]:


analysis1st.base_dir


# In[11]:


analysis1st.run('MultiProc', plugin_args={'n_procs': 22})


# In[12]:


get_ipython().system('tree /mnt/data/Project/DCM/BIDS/derivatives/Nipype/working_dir')


# In[ ]:




