import os
import numpy as np
import pandas as pd
from nilearn import masking, image
from scipy.stats import ttest_1samp, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game2_fmri>=0.5')
pid = data['Participant_ID'].to_list()
print(len(pid))

# set camp
cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/distance_spat/Setall/6fold/{}/zmap/M2xdistance_zmap.nii.gz'

# set roi
roi = image.load_img(
    r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/distance_spat/HC_ROI.nii.gz')
# roi = image.load_img(r'/mnt/workdir/DCM/docs/Mask/Park_Grid_ROI/ECl_roi.nii')
roi_thr_bin = image.binarize_img(roi, 0)

# extract mean activity
subs_mean_mean_activity = []
for sub_id in pid:
    sub_cmap = cmap_template.format(sub_id)
    mean_activity = np.mean(masking.apply_mask(sub_cmap, roi_thr_bin))
    subs_mean_mean_activity.append(mean_activity)

# %%  mean
# statistical test
t_statistic, p_value = ttest_1samp(subs_mean_mean_activity, 0)
print('t:', t_statistic)
print('p:', p_value)
g = sns.barplot(data=subs_mean_mean_activity, errorbar='sd', width=0.2)  # plot

# %%  correlation
covary_variable = data['Age'].to_list()
r, p = pearsonr(subs_mean_mean_activity, covary_variable)

# plot
g2 = sns.jointplot(x=covary_variable, y=subs_mean_mean_activity,
                  kind="reg", truncate=False,
                  xlim=(6, 30),  # ylim=(0, 1.05),
                  color="b", height=6)

# move overall title up
g2.set_axis_labels('Age', 'Z statistic', size=20)
g2.fig.subplots_adjust(top=0.92)
if p < 0.001:
    g2.fig.suptitle('r:{}  p<0.001'.format(round(r, 2)), size=20)
else:
    g2.fig.suptitle('r:{}, p:{}'.format(round(r, 2), round(p, 3)), size=20)
# g2.savefig(
#    r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/mPFC_covary_result',
#    dpi=300)

#%%
sns.lineplot(x=covary_variable,y=subs_mean_mean_activity)

# glm
from statsmodels.formula.api import glm
covary_variable = data['Age'].to_list()
mean = np.mean(covary_variable)
covary_variable_demean = [c-mean for c in covary_variable]
data = pd.DataFrame({'Age':covary_variable_demean,'z':subs_mean_mean_activity})
model = glm('z ~ Age',data=data).fit()
print(model.summary())

# %%
# set two roi and plot in one figure
# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
pid = data['Participant_ID'].to_list()

# set camp
cmap_template = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/{}/zstats_0011.nii.gz'

# set roi
roi1 = image.load_img(
    r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/EC_Peak_ROI.nii.gz')
roi2 = image.load_img(
    r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/PCC_Peak_ROI.nii.gz')

# extract mean activity
roi1_acts = []
roi2_acts = []
for sub_id in pid:
    print(sub_id)
    sub_cmap = cmap_template.format(sub_id)
    act1 = np.mean(masking.apply_mask(sub_cmap, roi1))
    act2 = np.mean(masking.apply_mask(sub_cmap, roi2))
    roi1_acts.append(act1)
    roi2_acts.append(act2)

age = data['Age']
# %%
red = (207 / 254, 113 / 255, 117 / 255)
blue = (112 / 255, 142 / 255, 191 / 255)
fig, ax = plt.subplots(figsize=(12,12))
plt.scatter(age, roi1_acts, color=red, label='EC', s=50., alpha=1)
#plt.scatter(age, roi2_acts, color=blue, label='PCC', s=50., alpha=1)

# Fit a linear regression line for each type of value

# slope1, intercept1 = np.polyfit(age, roi1_acts, 1)
# slope2, intercept2 = np.polyfit(age, roi2_acts, 1)
# x = np.arange(7.5, 30, 0.1)
# plt.plot(x, slope1 * x + intercept1, c=red, alpha=1)
# plt.plot(x, slope2 * x + intercept2, c=blue, alpha=1)
sns.regplot(x=age, y=roi1_acts,color=red)
#sns.regplot(x=age, y=roi2_acts,color=blue)
# Set the plot labels and legend
ax.set_xlabel('Age', fontsize=35)
ax.set_ylabel('Z statistic', fontsize=35)
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_xticks([10, 15, 20, 25])
ax.tick_params(axis='both', which='major', labelsize=30)
#ax.legend(fontsize=30)

# Show the plot

#plt.text(7.2, 3.7, 'mPFC r: 0.28 p<0.001 ', color=red, fontsize=25,fontweight='bold')
#plt.text(19, 3.7,'PCC  r:-0.26 p<0.001', color=blue, fontsize=25,fontweight='bold')
plt.text(12.5, 2.6, 'EC r: 0.31 p<0.001 ', color=red, fontsize=27,fontweight='bold')
#plt.text(19, 3.7,'PCC  r:-0.26 p<0.001', color=blue, fontsize=25,fontweight='bold')
#plt.title("mPFC r: 0.28 p<0.001   PCC r:-0.26 p<0.001",size=30)
plt.savefig(r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/EC_covary_result.png',dpi=300)
plt.show()