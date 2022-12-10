# plot the hot map for correct trials

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from analysis.mri.event.game1_event import Game1EV

sns.set_theme(style="darkgrid")


class Game1Map(Game1EV):
    def __init__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)
        self.game1_dformat()

    def sub_map_count(self):
        sub_map = np.zeros((5, 5))
        for index, row in self.behData.iterrows():
            pic1_ap = int(row['pic1_ap'])
            pic1_dp = int(row['pic1_dp'])

            pic2_ap = int(row['pic2_ap'])
            pic2_dp = int(row['pic2_dp'])

            sub_map[pic1_ap - 1, pic1_dp - 1] += 1
            sub_map[pic2_ap - 1, pic2_dp - 1] += 1
        return sub_map

    def sub_map_corr_count(self):
        sub_map = np.zeros((5, 5))
        trials_corr, acc = self.label_trial_corr()
        for (index, row), corr_label in zip(self.behData.iterrows(), trials_corr):
            if corr_label:
                pic1_ap = int(row['pic1_ap'])
                pic1_dp = int(row['pic1_dp'])

                pic2_ap = int(row['pic2_ap'])
                pic2_dp = int(row['pic2_dp'])

                sub_map[pic1_ap - 1, pic1_dp - 1] += 1
                sub_map[pic2_ap - 1, pic2_dp - 1] += 1
        return sub_map


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
#data = participants_data.query('game1_acc<0.7')
subjects_list = data['Participant_ID'].to_list()
behFile_template = r'/mnt/workdir/DCM/sourcedata/{}/Behaviour/fmri_task-game1/{}_task-game1_run-{}.csv'

sub_map = np.zeros((5, 5))
sub_corr_map = np.zeros((5, 5))
for sub_id in subjects_list:
    for run_id in range(1, 7):
        behFile = behFile_template.format(sub_id.replace('-','_'), sub_id, run_id)
        game1map = Game1Map(behFile)
        sub_map += game1map.sub_map_count()
        sub_corr_map += game1map.sub_map_corr_count()

#%%
# plot the heat map
sns.set_theme()
sub_corr_map = sub_corr_map/sub_corr_map.sum()
f, ax = plt.subplots(figsize=(8,7))
sns.heatmap(sub_corr_map,annot=True, linewidths=.5, ax=ax,cmap='coolwarm')
plt.title("The map of correct trials",size=20)

sub_map = sub_map/sub_map.sum()
f, ax = plt.subplots(figsize=(8,7))
sns.heatmap(sub_map,annot=True, linewidths=.5, ax=ax,cmap='coolwarm')
plt.title("The map of all trials",size=20)

diff_map = sub_corr_map - sub_map
f, ax = plt.subplots(figsize=(8,7))
sns.heatmap(diff_map,annot=True, linewidths=.5, ax=ax,cmap='coolwarm')
plt.title("The difference map",size=20)