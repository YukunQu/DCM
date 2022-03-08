# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 23:13:37 2021

@author: QYK
"""

# detect marker 
import os
import mne 
import numpy as np


fif_file_path = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_012/NeuroData/MEG/run4.fif'
raw = mne.io.read_raw_fif(fif_file_path,allow_maxshield=True)
raw.copy().pick_channels(ch_names=['STI101']).plot(start=1, duration=100)

detect_effective_channel = True
if detect_effective_channel:
    ch_names = raw.ch_names
    for ch_name in ch_names:
        events = mne.find_events(raw, stim_channel=ch_name)
        print(ch_name,':',len(events))  # show the first 5
    
# events_mark101 = mne.find_events(raw, stim_channel='STI101')
# #events_mark001 = mne.find_events(raw, stim_channel='STI001')
# #events_mark002 = mne.find_events(raw, stim_channel='STI002')
# #events = np.concatenate((events_mark101,events_mark001,events_mark002))
# event_dict = {'pic1': 1, 'pic2': 2, 'decision': 3}

ch_marker = mne.find_events(raw, stim_channel='STI101')
#ch_marker = mne.find_events(raw)
pic2_pic1_time_diff = []
dec_pic2_time_diff = []
decision_time = []

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
    if (nexttime_marker == 7) and (ontime_marker == 3):
        decision_time.append(time_diff)

for t1,t2,t3 in zip(pic2_pic1_time_diff,dec_pic2_time_diff,decision_time):
    print('Pic2- Pic1 time:',round(t1/1000,3))
    print('Decision time - Pic2 time:',round(t2/1000,3))
    print('Decision time:',round(t3/1000,3))


    