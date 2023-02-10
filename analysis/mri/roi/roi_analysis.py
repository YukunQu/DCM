import os
import numpy as np
import pandas as pd
from nilearn import masking,image
from scipy.stats import ttest_1samp,pearsonr
import seaborn as sns
sns.set_theme(style="whitegrid")

# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
pid = data['Participant_ID'].to_list()

# set camp
cmap_template = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/{}/zstats_0011.nii.gz'

# set roi
roi = image.load_img(r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/mPFC_Peak_ROI.nii.gz')
#roi = image.load_img(r'/mnt/workdir/DCM/docs/Mask/Park_Grid_ROI/ECl_roi.nii')
roi_thr_bin = image.binarize_img(roi,0)

# extract mean activity
subs_mean_mean_activity = []
for sub_id in pid:
    sub_cmap = cmap_template.format(sub_id)
    mean_activity = np.mean(masking.apply_mask(sub_cmap, roi_thr_bin))
    subs_mean_mean_activity.append(mean_activity)

#%%  mean
# statistical test
t_statistic, p_value = ttest_1samp(subs_mean_mean_activity,0)
print('t:',t_statistic)
print('p:',p_value)
g = sns.barplot(data=subs_mean_mean_activity,errorbar='sd',width=0.2) # plot

#%%  correlation
covary_variable = data['Age'].to_list()
#covary_variable = [18 if c >18 else c for c in covary_variable]
r,p = pearsonr(subs_mean_mean_activity,covary_variable)

# plot
g2 = sns.jointplot(x=covary_variable, y=subs_mean_mean_activity,
                  kind="reg", truncate=False,
                  xlim=(6, 30),#ylim=(0, 1.05),
                  color="r", height=6)

#move overall title up
g2.set_axis_labels('Age', 'Z statistic',size=20)
g2.fig.subplots_adjust(top=0.92)
if p < 0.001:
    g2.fig.suptitle('r:{}  p<0.001'.format(round(r,3)),size=20)
else:
    g2.fig.suptitle('r:{}, p:{}'.format(round(r,3),round(p,3)),size=20)
g2.savefig(r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/mPFC_covary_result',dpi=300)
