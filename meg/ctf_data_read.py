# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 23:54:35 2021

@author: QYK
"""

import mne

meg_filepath = r'D:\Data\Development_Cognitive_Map\BIDS\sourcedata\sub_003\NeuroData\MEG\TEST_G31BNU_20211011_01.ds'
raw = mne.io.read_raw_ctf(meg_filepath)
raw.copy().pick_channels(ch_names=['UPPT001']).plot(start=1, duration=20)
#%%
# find events
ch_names = raw.ch_names
#%%
for ch_name in ch_names:
    events = mne.find_events(raw, stim_channel=ch_name)
    print(ch_name,':',len(events)) 
#%% 
ch_marker = mne.find_events(raw, stim_channel='UPPT001')
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
    print('Pic2- Pic1 time:',round(t1/1200,5),"; ",'Decision time - Pic2 time:',round(t2/1200,5))

