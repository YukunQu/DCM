import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
import matplotlib.pyplot as plt


# generate different duration events
onset = [1]
trial_type = ['distance']
duration = [1]
event1 = pd.DataFrame({'onset':onset,'trial_type':trial_type,
                       'duration':duration,'modulation':1})

onset = [6]
trial_type = ['value']
duration = [1.5]
event2 = pd.DataFrame({'onset':onset,'trial_type':trial_type,
                       'duration':duration,'modulation':1})

events = pd.concat([event1,event2],axis=0)
frame_times= 0.5 * (np.arange(40))
design_matrix = make_first_level_design_matrix(frame_times,events=events,
                                               hrf_model='glover')
distance = design_matrix.loc[:,'distance']
value = design_matrix.loc[:,'value']

fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(distance,label='distance')
plt.plot(value,label='value')
colors = sns.color_palette()
ax.axvline(1, color=colors[0], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
ax.axvline(6, color=colors[1], linestyle='dashed', linewidth=1, alpha=0.5,zorder=0)
plt.legend()
plt.show()
