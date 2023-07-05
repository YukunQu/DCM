import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
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

# parameters
def generate_skewed_data(num_samples, skewness):
    # generate skewed random values
    random_values = skewnorm.rvs(a = skewness, size=num_samples)

    # scale and shift distribution to match desired mean and bounds
    random_values = (random_values - np.min(random_values))
    random_values = 0.5 + 2.5 * (random_values / np.max(random_values))
    mean_adjusted = np.mean(random_values)

    # shift distribution to match desired mean
    random_values = random_values + (1.5 - mean_adjusted)
    return random_values


# def generate_sample_mod(map_size=5,num=1000):
#     # generate the sample modulator
#
#     return distance,value


def generate_Ntrial_activity(onset,trial_type,duration,modulation):
    trial_num = len(onset)
    trials_act = np.zeros((trial_num,100))
    for i,(o,t,d,m) in enumerate(zip(onset,trial_type,duration,modulation)):
        events = pd.DataFrame({'onset':[o],'trial_type':[t],
                               'duration':[d],'modulation':[m]})
        frame_times= 0.2 * (np.arange(100))
        act = make_first_level_design_matrix(frame_times,events=events,
                                             hrf_model='spm')[trial_type[0]]
        trials_act[i,:] = act.values
    return trials_act


# generate distance effect * 1000 trials
onset = [1] * 100
trial_type = ['distance'] * 100
duration = [3] * 100
modulation = np.random.uniform(np.sqrt(2),4*np.sqrt(2),100) * 0.7
distance_act = generate_Ntrial_activity(onset,trial_type,duration,modulation)

# generate value effect * 1000 trials
x= [np.random.uniform(1,4) for i in range(100)]
onset = [1+2.5+i for i in x]
trial_type = ['value'] * 100
duration = generate_skewed_data(100, 3)
modulation = np.random.uniform(1,4,100) *2.2
value_act = generate_Ntrial_activity(onset,trial_type,duration,modulation)

#%%
# Convert the 2D arrays into DataFrames and reshape for seaborn
value_df = pd.DataFrame(value_act).melt(var_name='Time', value_name='Activity')
value_df['Type'] = 'Value'

distance_df = pd.DataFrame(distance_act).melt(var_name='Time', value_name='Activity')
distance_df['Type'] = 'Distance'

# Combine the two dataframes
combined_df = pd.concat([distance_df,value_df])
combined_df['Time'] = combined_df['Time'] * 0.2

# Create the lineplot with seaborn
fig, ax = plt.subplots(figsize=(7, 5))
colors = sns.color_palette('deep')
ax.axhline(0, color='black', linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(1, color=colors[0], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(6, color=colors[1], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
sns.lineplot(data=combined_df, x='Time', y='Activity', hue='Type', errorbar='se',palette=colors[:2],ax=ax)
# set y title
ax.set_ylabel('Mean activity (a.u.)')
# Set the y limit
ax.set_xlim(0, 20)
ax.set_ylim(-1, 2)
# Set y ticks
ax.set_xticks([0 ,5, 10, 15, 20])
ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
plt.title('Activity Over Time for Value and Distance')
# Add tick lines to the bottom and left spines
ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='out')

ax.tick_params(axis='y', which='both', left=True, right=False, direction='out')
savepath = r'/mnt/workdir/DCM/Result/paper/figure3/figure3_new/periplot_simulation.pdf'
plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300,transparent=True)
plt.show()