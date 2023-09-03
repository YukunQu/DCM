#%%
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import linregress
from nilearn import image
from nilearn import masking
import pingouin as pg

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game2_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()

hemis = ['lh','rh']
rois = ['EC','mPFC']
stats_names = ['thickness','volume']
hemi = 'lh'
roi = 'mPFC'
stats_name = 'thickness'

# load cognitive map's representation
#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/grid_rsa_corr_trials/Setall/6fold/{}/rsa/rsa_zscore_img_coarse_6fold.nii.gz'
cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/distance_spct/Setall/6fold/{}/zmap/distance_zmap.nii.gz'
# mask  = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz')
# mask = image.binarize_img(mask,5)
mask = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/VMPFC/VMPFC_MNI152NL_new.nii.gz')

# get activity in ROI
subs_cmap_list = [cmap_template.format(sub_id) for sub_id in subject_list]
subs_mean_activity = np.mean(masking.apply_mask(subs_cmap_list, mask),axis=1)
data['grid-like code'] = subs_mean_activity

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i, hemi in enumerate(hemis):
    for j, roi in enumerate(rois):
        outdir = f'/mnt/workdir/DCM/BIDS/derivatives/freesurfer_stats/{stats_name}/{hemi}.{roi}'
        stats = []
        for subjid in subject_list:
            filename = f'{outdir}/segstats-{subjid}.txt'

            with open(filename, 'r') as file:
                lines = file.readlines()

            last_line = lines[-1]
            values = last_line.split()
            mean_value = float(values[5])
            stats.append(mean_value)

        data[stats_name] = stats
        r, p = pearsonr(data['game2_test_acc'], data[stats_name])

        print(stats_name,f':{hemi} {roi}')
        p_corr =pg.partial_corr(data,'game2_test_acc', stats_name, covar=['game1_acc'], method='pearson')
        print('r:',round(p_corr['r'][0],3),'p:',round(p_corr['p-val'][0],3))

        ax = axes[i, j]
        data.plot.scatter(x='game2_test_acc', y=stats_name, ax=ax,color='red',alpha=0.3)
        slope, intercept, r_value, p_value, std_err = linregress(data['game2_test_acc'], data[stats_name])
        ax.plot(data['game2_test_acc'], slope*data['game2_test_acc'] + intercept, color='darkred', alpha=0.5)
        ax.set_title(f'{hemi} {roi} (r={r:.2f}, p={p:.2f})', fontsize=16)
        ax.set_xlabel('Game 2 Accuracy', fontsize=14)
        ax.set_ylabel(stats_name, fontsize=14)

plt.tight_layout()
plt.show()

#%%

import pandas as pd

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
subject_list = data['Participant_ID'].to_list()

hemis = ['lh', 'rh']
rois = ['EC', 'mPFC']
stats_names = ['thickness', 'volume']

results = []

for hemi in hemis:
    for roi in rois:
        for stats_name in stats_names:
            outdir = f'/mnt/workdir/DCM/BIDS/derivatives/freesurfer_stats/{stats_name}/{hemi}.{roi}'
            stats = []
            for subjid in subject_list:
                filename = f'{outdir}/segstats-{subjid}.txt'

                with open(filename, 'r') as file:
                    lines = file.readlines()

                last_line = lines[-1]
                values = last_line.split()
                mean_value = float(values[5])
                stats.append(mean_value)

            data[f'{hemi}_{roi}_{stats_name}'] = mean_value


data.to_csv('game1_acc_subject_stats.csv', index=False)


#%%
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import linregress
from nilearn import image
from nilearn import masking

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()

hemis = ['lh','rh']
rois = ['mPFC']
stats_names = ['volume','thickness']

# load cognitive map's representation
#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/grid_rsa_corr_trials/Setall/fold/{}/rsa/rsa_zscore_img_coarse_6fold.nii.gz'
#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_train_hexagon_spct/Setall/6fold/{}/zmap/hexagon_zmap.nii.gz'
#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_test_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_even_zmap.nii.gz'
#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/cv_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
#cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/cv_test_dmPFC_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
#cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/cv_mpfc_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
# mask  = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz')
# mask = image.binarize_img(mask,5)

cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/distance_spct/Setall/6fold/{}/zmap/distance_zmap.nii.gz'
mask = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/VMPFC/VMPFC_MNI152NL_new.nii.gz')

# get activity in ROI
subs_cmap_list = [cmap_template.format(sub_id) for sub_id in subject_list]
subs_mean_activity = np.max(masking.apply_mask(subs_cmap_list, mask),axis=1)
representation_name = 'distance code(Game1)'
data[representation_name] = subs_mean_activity

fig, axes = plt.subplots(2, figsize=(5, 10))

for i,stats_name in enumerate(stats_names):
    for roi in rois:
        stats = []
        for subjid in subject_list:
            mean_hemi_value = 0
            for hemi in hemis:
                outdir = f'/mnt/workdir/DCM/BIDS/derivatives/freesurfer_stats/{stats_name}/{hemi}.{roi}'
                filename = f'{outdir}/segstats-{subjid}.txt'
                with open(filename, 'r') as file:
                    lines = file.readlines()

                last_line = lines[-1]
                values = last_line.split()
                mean_hemi_value += float(values[5])

            mean_hemi_value = mean_hemi_value/2
            stats.append(mean_hemi_value)

        data[stats_name] = stats
        r, p = pearsonr(data[representation_name], data[stats_name])

        ax = axes[i]
        if stats_name == 'thickness':
            color = 'red'
            data.plot.scatter(x=representation_name, y=stats_name, ax=ax,color=color,alpha=0.3)
        elif stats_name == 'volume':
            color = 'blue'
            data.plot.scatter(x=representation_name, y=stats_name, ax=ax,alpha=0.8)
        slope, intercept, r_value, p_value, std_err = linregress(data[representation_name], data[stats_name])
        ax.plot(data[representation_name], slope*data[representation_name] + intercept, color=f'dark{color}', alpha=0.5)
        ax.set_title(f'{roi} (r={r:.2f}, p={p:.2f})', fontsize=16)
        ax.set_xlabel(representation_name, fontsize=16)
        ax.set_ylabel(stats_name, fontsize=16)

plt.tight_layout()
plt.show()