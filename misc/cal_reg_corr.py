# calculate the correlation distribution between two regressors' modulation

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# specify subjects
participants_data = pd.read_csv('/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
data = participants_data.query('game1_fmri>=0.5')
sub_list = data['Participant_ID'].to_list()

event1_tmp = r'/mnt/data/DCM/result_backup/2023.5.14/Events/game1/distance_spct/{}/6fold/{}_task-game1_run-{}_events.tsv'
event2_tmp = r'/mnt/data/DCM/result_backup/2023.5.14/Events/game1/value_spct/{}/6fold/{}_task-game1_run-{}_events.tsv'
corr_list = []
for sub in sub_list:
    mod1 = []
    mod2 = []
    for run_id in range(1,7):
        event1_path = event1_tmp.format(sub,sub,run_id)
        event2_path = event2_tmp.format(sub,sub,run_id)
        event1 = pd.read_csv(event1_path,sep='\t')
        event2 = pd.read_csv(event2_path,sep='\t')
        # corr = pearsonr(event1[event1['trial_type']=='distance']['modulation'].to_list(),
        #                  event2[event2['trial_type']=='value']['modulation'].to_list())[0]
        mod1.extend(event1[event1['trial_type'] == 'distance']['modulation'].to_list())
        mod2.extend(event2[event2['trial_type'] == 'value']['modulation'].to_list())
    corr = pearsonr(mod1,mod2)[0]
    corr_list.append(corr)

plt.hist(corr_list)
print(np.mean(corr_list),np.std(corr_list))
ttest_1samp(corr_list,0)