import numpy as np
import pandas as pd
from nilearn import masking,image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
#%%

# Compare covariate effect in different ROI with different contrast
covariate = 'age'

zmap_template = r'/mnt/data/DCM/result_backup/2022.11.22/game1_hp_noICA/separate_hexagon_2phases_correct_trials/Setall/group/covariates' \
                r'/{}/2ndLevel/_contrast_id_{}/spmT_0002.nii'

# roi
rois = {'EC':r'/mnt/workdir/DCM/docs/Reference/Mask/EC_ROI/volume/EC-thr25-2mm.nii.gz',
       'mPFC':r'/mnt/workdir/DCM/docs/Reference/Mask/Park_Grid_ROI/mPFC_roi.nii'}

# contrast
contrast = ['ZF_0003']
contrast_names = ['M2','Decision','Average']

mni_template = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
#%%
df1 = pd.DataFrame()
df2 = pd.DataFrame()
for roi_name,roi_path in rois.items():
    for i,con in enumerate(contrast):
        stat_map = zmap_template.format(covariate,con)
        roi_resampled = image.resample_to_img(roi_path,mni_template,'nearest')
        roi_values = masking.apply_mask(stat_map, roi_resampled)
        voxel_num_above_thr = sum(roi_values>=2.3)
        df1 = df1.append({'covariate':covariate,'contrast':contrast_names[i],'voxel_num':voxel_num_above_thr,
                          'roi':roi_name},ignore_index=True)
        for v in roi_values:
            df2 = df2.append({'contrast':contrast_names[i],'roi':roi_name,'covariate':covariate,'value':v},ignore_index=True)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(28,28))
#g = sns.catplot(data=df1, kind="bar",x="contrast", y="voxel_num",hue='roi',palette="dark", alpha=.6, height=6)
g =sns.catplot(data=df2,kind='bar',x="contrast",y="value",hue="roi",palette="dark", alpha=.6, height=6)
g.set(ylim=(0,2),title="hexagonal effect covary with age.")

#%%
# mean effect of high performance subjects across different phase in EC
mni_template = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
zmap_template = r'/mnt/data/DCM/result_backup/2022.11.22/game1_hp/separate_hexagon_2phases_correct_trials/Setall/group/hp/2ndLevel/_contrast_id_{}/spmT_0001.nii'
roi_path = r'/mnt/workdir/DCM/docs/Reference/Mask/EC_ROI/volume/EC-thr50-2mm.nii.gz'

contrast = ['ZF_0005','ZF_0006','ZF_0011']
contrast_names = ['M2','Decision','Average']

df1 = pd.DataFrame()
df2 = pd.DataFrame()
for i,con in enumerate(contrast):
    zmap = zmap_template.format(con)
    roi_resampled = image.resample_to_img(roi_path,mni_template,'nearest')
    roi_zvalues = masking.apply_mask(zmap, roi_resampled)
    voxel_num_above_thr = sum(roi_zvalues >= 2.3)
    df1 = df1.append({'contrast':contrast_names[i],'voxel_num':voxel_num_above_thr,
                      'mean':roi_zvalues.mean()},ignore_index=True)
    for v in roi_zvalues:
        df2 = df2.append({'contrast':contrast_names[i],'zvalue':v},ignore_index=True)
sns.set_theme(style="whitegrid")


plt.figure(figsize=(10,20))
g = sns.catplot(data=df2,kind='bar',x="contrast",y="zvalue",palette="dark", alpha=.6, height=6)
g.set(ylim=(-1,0.5))
#%%
# chart the individual difference between different phases
import numpy as np
import pandas as pd
from nilearn import masking,image

import matplotlib.pyplot as plt
import seaborn as sns

# roi
roi = r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_R_prob-80.nii.gz'
mni_template = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
roi_resampled = image.resample_to_img(roi,mni_template,'nearest')

# subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
data = data.query("(game1_acc>=0.7)and(Age>=13)")
subjects = data['Participant_ID'].to_list()
age_list = data['Age'].to_list()
# contrast
contrast = ['0005','0014','0006','0011','0015']
contrast_names = ['M2','Planning','Decision','Average','F-test']

zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/cv_test_dmPFC/Setall/6fold/{}/ZT_0002.nii'

df = pd.DataFrame()
for sub,age in zip(subjects,age_list):
        zmap = zmap_template.format(sub)
        roi_zvalues = masking.apply_mask(zmap,roi_resampled)
        df = df.append({'subject':sub,'Age':age,'mean_hexagonal modulation':roi_zvalues.mean()}, ignore_index=True)


sns.set_theme(style="darkgrid")
plt.figure(figsize=(7,7))
plt.scatter(data=df, x="Age", y="mean_hexagonal modulation")

#%%
# roi
roi = r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_R_prob-80.nii.gz'
mni_template = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
roi_resampled = image.resample_to_img(roi,mni_template,'nearest')

# subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
data = data.query("(game1_acc>=0.7)and(Age>=13)")
subjects = data['Participant_ID'].to_list()
acc_list = data['game1_acc'].to_list()


zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/cv_test_dmPFC/Setall/6fold/{}/ZT_0002.nii'

df = pd.DataFrame()
for sub,acc in zip(subjects,acc_list):
    zmap = zmap_template.format(sub)
    roi_zvalues = masking.apply_mask(zmap,roi_resampled)
    df = df.append({'subject':sub,'Acc':acc,'mean_hexagonal modulation':roi_zvalues.mean()}, ignore_index=True)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(7,7))
plt.scatter(data=df, x="Acc", y="mean_hexagonal modulation")