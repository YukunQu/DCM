# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:24:49 2021

@author: QYK
"""
import random
import numpy as np
from os.path import join
import pandas as pd

import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix

#%%
# simulation parameters
omegas = range(-29,30,5) # 根据Erie 的代码重新改范围
omega = random.choice(omegas)
tr = 3 # repetition time is 1 second
n_scans = 804 # the acquisition comprises 128 scans
frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times

# generate a simulation serial data from events file
events_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-003\events'
events_path = join(events_dir,'sub-003_task-game1_run-4to9_events.tsv')
events = pd.read_csv(events_path,sep="\t")
events = events[events['trial_type'] == 'M2_sin']
events = events[['onset','duration','angle']]
events['trial_type'] = 'simulation'
events['modulation'] = 1 + 1*np.cos(np.deg2rad(6*(events['angle'] - omega)))
events = events.drop('angle',axis=1)
design_matrix = make_first_level_design_matrix(frame_times, events, drift_model='polynomial', drift_order=0,hrf_model='pyspm')
simulation_data = np.array(design_matrix['simulation'])
#%%
# creat a brain activity image
func = nib.load(r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-003\func/sub-003_task-game1_run-4to9_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
dim =func.shape
img = np.zeros(dim)
affine = func.affine

mask = nib.load(r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-003\mask/sub-003_-MNI152_T1w_Hippocampus_roi.nii.gz')
mask = mask.get_fdata()[:,:,:,0]

for i in range(804):
    img_slice = img[:,:,:,i]
    img_slice[mask>0] = simulation_data[i]*100 + np.random.rand() * 20
    img[:,:,:,i] = img_slice

sim_brain_act =nib.Nifti1Image(img, affine=affine)
sim_brain_act.to_filename(r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-003\func\simulation/simulation_brain_activation.nii.gz')