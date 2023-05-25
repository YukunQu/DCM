import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from scipy.signal import deconvolve


def p_to_z(p):
    from scipy.stats import norm
    return norm.ppf(1 - p)


# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'(game1_fmri>=0.5)and(game1_acc>=0.9)')
pid = data['Participant_ID'].to_list()
subjects = [p.split('-')[-1] for p in pid]

# load data
results = pd.DataFrame(columns=['subj', 'run', 'time_point', 'beta', 'p'])
for subj in subjects:
    sub_res = pd.read_csv(f'/mnt/data/DCM/derivatives/peri_event_analysis/EC/sub-{subj}_peri_event_analysis.csv')
    results = results.append(sub_res)

# results = pd.DataFrame(columns=['subj', 'run', 'time_point', 'beta', 'p'])
# for subj in subjects:
#     sub_res = pd.read_csv(f'/mnt/data/DCM/derivatives/peri_event_analysis/vmPFC_test/sub-{subj}_peri_event_analysis.csv')
#     results = results.append(sub_res)
#
# results2 = pd.DataFrame(columns=['subj', 'run', 'time_point', 'beta', 'p'])
# for subj in subjects:
#     sub_res = pd.read_csv(f'/mnt/data/DCM/derivatives/peri_event_analysis/dmPFC_test/sub-{subj}_peri_event_analysis.csv')
#     results2 = results2.append(sub_res)

results['time_point'] = results['time_point']/5
results['z'] = p_to_z(results['f_p'])
# results2['time_point'] = results2['time_point']/5
# results2['distance_z'] = p_to_z(results2['distance_p'])
# results['value_z'] = p_to_z(results['value_p'])

#%%
# plot peri-stimulus time course for all subjects
fig, ax = plt.subplots(figsize=(10, 5))
# add a dashed horizontal line at y=0
# get the colors used by seaborn
colors = sns.color_palette()
#ax.axhline(0, color='black', linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(1, color=colors[0], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(6, color=colors[1], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
sns.lineplot(x='time_point', y='z', data=results, ax=ax,errorbar='se',label='hexagon')
#sns.lineplot(x='time_point', y='value_beta', data=results, ax=ax,errorbar='se',label='value')
ax.legend()
# set y title
ax.set_ylabel('Z')
