# calculate the correlation distribution between two regressors' modulation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# specify subjects
participants_data = pd.read_csv('/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
data = participants_data.query('game2_fmri>=0.5')
sub_list = data['Participant_ID'].to_list()

event1_tmp = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/hexagon_spct/{}/6fold/{}_task-game2_run-{}_events.tsv'
event2_tmp = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/hexagon_center_spct/{}/6fold/{}_task-game2_run-{}_events.tsv'
corr_list = []
for sub in sub_list:
    for run_id in [1,2]:
        event1_path = event1_tmp.format(sub,sub,run_id)
        event2_path = event2_tmp.format(sub,sub,run_id)
        event1 = pd.read_csv(event1_path,sep='\t')
        event2 = pd.read_csv(event2_path,sep='\t')
        corr = event1[event1['trial_type']=='cos']['modulation'].corr(event2[event2['trial_type']=='cos']['modulation'])
        corr_list.append(corr)

plt.hist(corr_list)
print(np.mean(corr_list),np.std(corr_list))
