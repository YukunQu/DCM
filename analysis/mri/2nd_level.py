#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# In[2]:

# subject 
adult_list = ['010','018','023','024','027','029','033',
              '037','043','046','047','053','056','058','062']

ado_list =   ['016', '022', '031', '032', '036', '049', '050', '055', '059',
              '060', '061']

child_list = ['011', '012', '015', '017', '025', '040', '048', '063', '064']

high_performance_subjects = ['010','024','032','036','043','046','053','061','062'] # game1_acc > 0.75

sub_type = 'adult'

if sub_type == 'adult':
    subject_list  = adult_list
elif sub_type == 'adolescent':
    subject_list  = ado_list
elif sub_type == 'child':
    subject_list  = child_list
elif sub_type == 'hp':  
    subject_list  = high_performance_subjects
#%%
# data input and ouput
contrast_list = ['ZF_0004']
infosource = Node(IdentityInterface(fields=['contrast_id','subj_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list)]
infosource.inputs.subj_id = subject_list

# SelectFiles - to grab the data (alternativ to DataGrabber)
data_root = '/mnt/data/Project/DCM/BIDS/derivatives/Nipype'
templates = {'cons': pjoin(data_root, 'M2short/1stLevel_part2', 'sub-{subj_id}','{contrast_id}.nii')}

# Create SelectFiles node
selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                   name='selectfiles', iterfield=['subj_id'])

# Initiate DataSink node here
datasink = Node(DataSink(base_directory='/mnt/data/Project/DCM/BIDS/derivatives/Nipype/',
                         container='M2/M2_mod_sess2/{}'.format(sub_type)),
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
analysis2nd = Workflow(name='work_2nd_{}'.format(sub_type), base_dir='/mnt/data/Project/DCM/BIDS/derivatives/Nipype/working_dir')
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

# Create 2nd-level analysis output graph
analysis2nd.write_graph(graph2use='colored', format='png', simple_form=True)
# run 2nd analysis
analysis2nd.run('MultiProc', plugin_args={'n_procs': 40})