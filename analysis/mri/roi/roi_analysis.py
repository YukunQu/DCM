import os
import numpy as np
import pandas as pd
from nilearn import masking,image
from scipy.stats import ttest_1samp
import seaborn as sns
sns.set_theme(style="whitegrid")

# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>0.5')
pid = data['Participant_ID'].to_list()

# set camp
cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/distance/Setall/6fold/{}/con_0005.nii'

# set roi
roi = r'/mnt/workdir/DCM/docs/Mask/Park_Grid_ROI/HCl_roi.nii'
mni_template = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
roi_resampled = image.resample_to_img(roi, mni_template, 'nearest')

# extract mean activity
subs_mean_mean_activity = []
for sub_id in pid:
    sub_cmap = cmap_template.format(sub_id)
    mean_activity = np.mean(masking.apply_mask(sub_cmap, roi_resampled))
    subs_mean_mean_activity.append(mean_activity)

# statistical test
t_statistic, p_value = ttest_1samp(subs_mean_mean_activity,0)
print(t_statistic, p_value)

# plot
g = sns.catplot(data=subs_mean_mean_activity,errorbar='sd')