import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
sns.set_style('white')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})
# Set the default visibility of the spines
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True


def p_to_z(p):
    from scipy.stats import norm
    return norm.ppf(1-p)


# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
# Define age groups
bins = [8, 12, 17, 25]
labels = ['Children', 'Adolescent', 'Adults']
# Assign age groups
participants_data['Age_Group'] = pd.cut(participants_data['Age'], bins=bins, labels=labels, include_lowest=True)
data = participants_data.query(f'(game1_fmri>=0.5)')
pid = data['Participant_ID'].to_list()
subjects = [p.split('-')[-1] for p in pid]

# load data
results = pd.DataFrame(columns=['subj', 'run', 'time_point', 'beta', 'p'])
for subj in subjects:
    sub_res = pd.read_csv(f'/mnt/data/DCM/derivatives/peri_event_analysis/vmPFC/sub-{subj}_peri_event_analysis.csv')
    results = results.append(sub_res)

results2 = pd.DataFrame(columns=['subj', 'run', 'time_point', 'distance_beta','value_beta'])
for subj in subjects:
    sub_res = pd.read_csv(f'/mnt/data/DCM/derivatives/peri_event_analysis/vmPFC/sub-{subj}_peri_event_analysis.csv')
    results2 = results2.append(sub_res)

results['time_point'] = results['time_point']/5
results2['time_point'] = results2['time_point']/5

# Add age group information to the results dataframe
results['Age_Group'] = results['subj'].map(data.set_index('Participant_ID')['Age_Group'])
results2['Age_Group'] = results2['subj'].map(data.set_index('Participant_ID')['Age_Group'])

#%%
# plot peri-stimulus time course for all subjects
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='out')
ax.tick_params(axis='y', which='both', left=True, right=False, direction='out')
# get the colors used by seaborn
colors = sns.color_palette('bright')
# add a dashed horizontal line at y=0
ax.axhline(0, color='black', linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(1, color='#f07167', linestyle='dashed', linewidth=2, alpha=0.5,zorder=0)
ax.axvline(6, color='#FFBC6F', linestyle='dashed', linewidth=2, alpha=0.5,zorder=0)
sns.lineplot(x='time_point', y='value_beta', data=results2, ax=ax,errorbar='se',label='Value',alpha=1,color='#FFBC6F',zorder=1)
sns.lineplot(x='time_point', y='distance_beta', data=results, ax=ax,errorbar='se',label='Distance',color='#f07167',zorder=0)
ax.legend(loc='best',frameon=False,bbox_to_anchor=(1, 0.8))

# set y title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean activity (a.u.)')
# # Set the y limit
# ax.set_xlim(0, 20)
# ax.set_ylim(-2, 3.5)
# # Set y ticks
# ax.set_xticks([0 ,5, 10, 15, 20])
# ax.set_yticks([-2, -1, 0, 1, 2, 3])
# Set the y limit
ax.set_xlim(0, 20)
# ax.set_ylim(-1, 2)
# Set y ticks
ax.set_xticks([0 ,5, 10, 15, 20])
# ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
# Add tick lines to the bottom and left spines
ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='out')
ax.tick_params(axis='y', which='both', left=True, right=False, direction='out')
#savepath = r'/mnt/workdir/DCM/Result/paper/figure3/periplot_all_subjects.pdf'
#plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300,transparent=True)

#%%
# plot peri-stimulus time course for different age group of specified effect
# distance effect
fig, ax = plt.subplots(figsize=(7, 5))
#distance_colors = [sns.color_palette('pastel')[0],'dodgerblue','midnightblue']
distance_colors = ['silver','lightskyblue',sns.color_palette('deep')[0]]
ax.axhline(0, color='black', linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(1, color=colors[0], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(6, color=colors[1], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
sns.lineplot(x='time_point', y='distance_beta', data=results,hue='Age_Group', ax=ax,errorbar='se',
             palette=distance_colors,hue_order=labels) #
# set y title
ax.set_ylabel('Mean activity (a.u.)')
# Set the y limit
ax.set_xlim(0, 20)
ax.set_ylim(-1, 2)
# Set y ticks
ax.set_xticks([0,5,10,15,20])
ax.set_yticks([-2, -1, 0, 1, 2])
# Add tick lines to the bottom and left spines
ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='out')
ax.tick_params(axis='y', which='both', left=True, right=False, direction='out')
savepath = r'/mnt/workdir/DCM/Result/paper/figure3/figure3_new_planB/periplot_distance_different_age2.pdf'
plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300,transparent=True)


#%%
# value effect
fig, ax = plt.subplots(figsize=(7, 5))
value_colors = ['silver','darkorange','maroon']
ax.axhline(0, color='black', linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(1, color=colors[0], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(6, color=colors[1], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
sns.lineplot(x='time_point', y='value_beta', data=results2,hue='Age_Group', ax=ax,errorbar='se',
             palette=value_colors,hue_order=labels)
# set y title
ax.set_ylabel('Mean activity (a.u.)')

ax.set_xlim(0, 20)
ax.set_ylim(-2, 3.5)
# Set y ticks
ax.set_xticks([0,5,10,15,20])
ax.set_yticks([-2, -1, 0, 1, 2, 3])
# Add tick lines to the bottom and left spines
ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='out')
ax.tick_params(axis='y', which='both', left=True, right=False, direction='out')
savepath = r'/mnt/workdir/DCM/Result/paper/figure3/figure3_new_planB/periplot_value_different_age2.pdf'
plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300,transparent=True)

#%%
# do statistical inference
from scipy.stats import ttest_1samp,ttest_ind

X = np.zeros((len(subjects)*6,100))
for i,(index,time) in enumerate(results.groupby('time_point')):
    X[:,i] = time['value_beta']


# Perform t-test at each time point
t_values, p_values = ttest_1samp(X,0,axis=0)
for i,p in enumerate(p_values):
    if p < 0.05:
        print(i/5,p)


from mne.stats import permutation_cluster_1samp_test
import numpy as np

# apply permutation cluster 1-sample test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(X, n_permutations=1000, threshold=None, tail=0)

print(clusters[0][0]/5)
print(cluster_p_values)


#%%
# calculate the peak time for each subject
# convert data to numeric if needed
results['distance_beta'] = pd.to_numeric(results['distance_beta'])
results['value_beta'] = pd.to_numeric(results['value_beta'])
results2['distance_beta'] = pd.to_numeric(results2['distance_beta'])
results2['value_beta'] = pd.to_numeric(results2['value_beta'])

# calculate the mean of 'distance_beta' and 'value_beta' for each subject and time_point
results_avg = results.groupby(['subj', 'time_point'])[['distance_beta', 'value_beta']].mean().reset_index()
results2_avg = results2.groupby(['subj', 'time_point'])[['distance_beta', 'value_beta']].mean().reset_index()

def find_peak_time(X):
    # X: n_subjects x n_timepoints
    peak = np.argmax(X,axis=1)
    return peak

X = np.zeros((len(subjects)*6,100))
for i,(index,time) in enumerate(results.groupby('time_point')):
    X[:,i] = time['distance_beta']

peak_times = find_peak_time(X)/5
print(peak_times.mean())
np.save('/mnt/data/DCM/derivatives/peri_event_analysis/distance_adolescents_peak_times.npy', peak_times)

X = np.zeros((len(subjects)*6,100))
for i,(index,time) in enumerate(results2.groupby('time_point')):
    X[:,i] = time['value_beta']

peak_times = find_peak_time(X)/5
print(peak_times.mean())
np.save('/mnt/data/DCM/derivatives/peri_event_analysis/value_adolescents_peak_times.npy', peak_times)
