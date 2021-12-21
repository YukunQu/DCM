# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:50:08 2021

@author: QYK
"""

import os
import mne 
import numpy as np
from autoreject import get_rejection_threshold

fif_file_path = r'D:\Data\Development_Cognitive_Map\BIDS\sourcedata\sub_002\NeuroData\MEG/run1.fif'
raw = mne.io.read_raw_fif(fif_file_path,allow_maxshield=True)
raw.copy().pick_channels(ch_names=['STI101']).plot(start=1, duration=100)
#%%

#%%
# 
raw.plot_psd(fmax=50)
raw.plot(duration=30, n_channels=30)

#%%
# preprocess
# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)

#%%
# find events
ch_names = raw.ch_names
#%%
for ch_name in ch_names:
    events = mne.find_events(raw, stim_channel=ch_name)
    print(ch_name,':',len(events))  # show the first 5

#%%
# events_mark101 = mne.find_events(raw, stim_channel='STI101')
# #events_mark001 = mne.find_events(raw, stim_channel='STI001')
# #events_mark002 = mne.find_events(raw, stim_channel='STI002')
# #events = np.concatenate((events_mark101,events_mark001,events_mark002))
# event_dict = {'pic1': 1, 'pic2': 2, 'decision': 3}

ch_marker = mne.find_events(raw, stim_channel='STI101')
pic2_pic1_time_diff = []
dec_pic2_time_diff = []

for index in range(len(ch_marker)-1):
    ontime = ch_marker[index]
    nexttime = ch_marker[index+1]
    
    ontime_marker = ontime[-1]
    nexttime_marker = nexttime[-1]
    time_diff = nexttime[0] - ontime[0]
    if (nexttime_marker == 2) and (ontime_marker == 1):
        pic2_pic1_time_diff.append(time_diff)
    elif (nexttime_marker == 3) and (ontime_marker == 2):
        dec_pic2_time_diff.append(time_diff)

for t1,t2 in zip(pic2_pic1_time_diff,dec_pic2_time_diff):
    print('Pic2- Pic1 time:',round(t1/1000,3),";  ",'Decision time - Pic2 time:',round(t2/1000,3))
    
"""
epochs = mne.Epochs(raw, events_mark101, event_id=event_dict, tmin=-0.2, tmax=0.5, preload=True)

reject_criteria = get_rejection_threshold(epochs) 
epochs = mne.Epochs(raw, events_mark101, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)
"""
