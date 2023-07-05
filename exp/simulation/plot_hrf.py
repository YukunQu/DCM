# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:20:04 2021

@author: QYK
"""
import numpy as np
import pandas as pd
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
import matplotlib.pyplot as plt


# generate the simulation sequence
onset = np.linspace(0,10,10) * 16

# generate different duration events
trial_type = ['box car'] * len(onset)
duration = [26] * len(onset)
event1 = pd.DataFrame({'onset':onset,'trial_type':trial_type,
                       'duration':duration})

trial_type = ['stick']* len(onset)
duration = [0] * len(onset)
onset1 = [0]
trial_type = ['stick']* len(onset1)
duration = [0] * len(onset1)
event2 = pd.DataFrame({'onset':onset1,'trial_type':trial_type,
                       'duration':duration})

events = pd.concat([event1,event2],axis=0)
frame_times= 0.5 * (np.arange(133))
design_matrix = make_first_level_design_matrix(frame_times,events=events,
                                               hrf_model='spm')
box_car = design_matrix.loc[:100,'box_car']
stick = design_matrix.loc[:100,'stick']

#plt.plot(box_car)
plt.plot(stick)
plt.show()