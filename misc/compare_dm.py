from misc.load_spm import SPMfile
from analysis.mri.voxel_wise.nilearn.Ftest.game1_separate_phases_correct_trials import prepare_data
import nilearn.plotting as plotting

subj = '010'
# load spm dm
spm_mat = SPMfile(r'/mnt/data/DCM/result_backup/2022.11.27/game1/'
                  rf'separate_hexagon_2phases_correct_trials/Setall/6fold/sub-{subj}/SPM.mat')
spm_dm = spm_mat.design_matrix

#%%
# load nilearn dm
run_list = [1,2,3,4,5,6]
ifold = 6
configs = {'TR':3.0, 'task':'game1', 'glm_type':'separate_hexagon_2phases_correct_trials',
           'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
           'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
           'func_name': r'sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth8.nii',
           'events_name':r'sub-{}_task-game1_run-{}_events.tsv',
           'regressor_name':r'sub-{}_task-game1_run-{}_desc-confounds_timeseries.tsv'}
_, nilearn_dm = prepare_data(subj,run_list,ifold,configs,True)
#%%
# plot m1 for run1
plotting.plot_design_matrix(spm_dm.iloc[:160,:17])
# plot m1 for run1
plotting.plot_design_matrix(nilearn_dm.iloc[:160,:17])

#%%
from scipy.stats import pearsonr

for i,con_name in enumerate(nilearn_dm.iloc[:160,:17].columns):
    r,p = pearsonr(spm_dm.iloc[:160,i],nilearn_dm.iloc[:160,i])
    print(round(r,4),con_name)

#%%
import matplotlib.pyplot as plt

# Create a figure with one row and four columns
fig, axs = plt.subplots(4, 1, figsize=(15,10))

# Plot the dataframe in each subplot
for i, col in enumerate(spm_dm.iloc[:30,:4].columns):
    spm_dm.iloc[:30,:4][col].plot(ax=axs[i],color='blue',label='spm',title=col)

for i, col in enumerate(nilearn_dm.iloc[:30,:4].columns):
    nilearn_dm.iloc[:30,:4][col].plot(ax=axs[i],color='red',label='nilearn',title=col)
plt.legend()