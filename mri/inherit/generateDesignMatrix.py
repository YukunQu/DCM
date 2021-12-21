# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:03:38 2021

@author: QYK
"""

from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

sub_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\fmriprep\sub-003'
for run_id in range(4,10):
    # set paramters
    tr = 3 # repetition time is 1 second
    n_scans = 134 # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
    
    # import events file
    tsv_file = join(sub_dir,'events','new_events','sub-003_task-game1_run-{}_events_new.tsv'.format(run_id))
    events = pd.read_csv(tsv_file,sep="\t")
    
    # Next, we import 6 motion parameters
    # The 6 parameters correspond to three translations and three rotations 
    # describing rigid body motion
    
    add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound_file = join(sub_dir,'func','sub-003_task-game1_run-{}_desc-confounds_timeseries.tsv'.format(run_id))
    confound_factors = pd.read_csv(confound_file,sep="\t")
    motion = confound_factors[add_reg_names]
    
    # generate a design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times, events, drift_model=None,
        add_regs=motion, add_reg_names=add_reg_names, hrf_model='spm')
    dm_save_path = join(sub_dir,'events','new_events','sub-003_task-game1_run-{}_design_matrix_new.tsv'.format(run_id))
    #design_matrix.to_csv(dm_save_path,sep='\t',index=False)

#%%
plot_design_matrix(design_matrix)
plt.show()
