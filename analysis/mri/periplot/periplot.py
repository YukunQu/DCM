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
data = participants_data.query(f'(game1_fmri>=0.5)')
pid = data['Participant_ID'].to_list()
subjects = [p.split('-')[-1] for p in pid]

# load data
results = pd.DataFrame(columns=['subj', 'run', 'time_point', 'beta', 'p'])
for subj in subjects:
    sub_res = pd.read_csv(f'/mnt/data/DCM/derivatives/peri_event_analysis/dmPFC_test/sub-{subj}_peri_event_analysis.csv')
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
#results['z'] = p_to_z(results['f_p'])
# results2['time_point'] = results2['time_point']/5
results['distance_z'] = p_to_z(results['distance_p'])
results['value_z'] = p_to_z(results['value_p'])
# results2['distance_z'] = p_to_z(results2['distance_p'])
# results2['value_z'] = p_to_z(results2['value_p'])

#%%
# plot peri-stimulus time course for all subjects
fig, ax = plt.subplots(figsize=(10, 5))
# add a dashed horizontal line at y=0
# get the colors used by seaborn
colors = sns.color_palette()
ax.axhline(0, color='black', linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(1, color=colors[0], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(6, color=colors[1], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
sns.lineplot(x='time_point', y='distance_beta', data=results, ax=ax,errorbar='se',label='Distance')
sns.lineplot(x='time_point', y='value_', data=results, ax=ax,errorbar='se',label='Value',color=colors[1])
ax.legend()
# set y title
ax.set_ylabel('Z')

#%%
# statistical inference
from scipy.stats import ttest_1samp

X = np.zeros((len(subjects)*6,100))
for i,(index,time) in enumerate(results.groupby('time_point')):
    X[:,i] = time['distance_beta']
#%%
from scipy.stats import ttest_ind
# Perform t-test at each time point
t_values, p_values = ttest_1samp(X,0,axis=0)
for i,p in enumerate(p_values):
    if p<0.05:
        print(i/5,p)

#%% permutation test
from mne.stats import permutation_cluster_1samp_test
import numpy as np

# Apply permutation cluster 1-sample test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(X, n_permutations=1000, threshold=None, tail=0)

print(clusters[0][0]/5)
print(cluster_p_values)