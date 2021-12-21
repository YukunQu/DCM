# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:24:59 2021

@author: QYK
"""
import os
import pandas as pd 
from nilearn.glm.first_level import make_first_level_design_matrix

# concat the events

event_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-005\events'
event_all = []
for i,run_id in enumerate(range(1,10)):
    event_path = os.path.join(event_dir,'sub-005_task-game1_run-{}_events.tsv'.format(run_id))
    event = pd.read_csv(event_path,sep='\t')
    event['onset'] = event['onset'] + i*134*3
    event_all.append(event)
event_all = pd.concat(event_all)
save_path = os.path.join(event_dir,'sub-005_task-game1_run-all_events.tsv')
event_all.to_csv(save_path,sep='\t',index=False)

#%%
# concat the design matrix
dm_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-003\events\new_events'
dm_run4to9 = []
for i,run_id in enumerate(range(4,10)):
    dm_path = os.path.join(dm_dir,'sub-003_task-game1_run-{}_design_matrix_new.tsv'.format(run_id))
    dm = pd.read_csv(dm_path,sep='\t')
    dm_run4to9.append(dm)
dm_4to9 = pd.concat(dm_run4to9)
save_path = os.path.join(dm_dir,'sub-003_task-game1_run-4to9_design_matrix_new.tsv')
dm_4to9.to_csv(save_path,sep='\t',index=False)  

#%%
from nilearn.image import concat_imgs
# concat the fMRI data
func_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-005\func'
func_all = []
for i,run_id in enumerate(range(1,10)):
    dm_path = os.path.join(func_dir,'sub-005_task-game1_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(run_id))
    func_all.append(dm_path)
func_all = concat_imgs(func_all)
save_path = os.path.join(func_dir,'sub-005_task-game1_run-all_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
func_all.to_filename(save_path)