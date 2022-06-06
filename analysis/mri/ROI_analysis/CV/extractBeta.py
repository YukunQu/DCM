import json
import os

import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img,resample_to_img
#%%
def extractBetas(subjects,roi,stats_dir,cmap):
    sub_cv_beta = pd.DataFrame(columns=['sub_id','ifold','set_id','amplitude'])
    folds = range(4,9)
    testset = ['Set1', 'Set2']
    for i in folds:
        ifold = str(i)+'fold'
        print(f"________{ifold} start____________")
        for sub in subjects:
            for set_id in testset:
                stats_map = os.path.join(stats_dir,set_id,ifold,sub,cmap)
                roi_img = load_img(roi)
                roi_img = resample_to_img(roi_img, stats_map,interpolation='nearest')
                betas = apply_mask(imgs=stats_map, mask_img=roi_img)
                betas[betas==0] = np.nan
                amplitude = np.nanmean(betas)
                tmp_data = {'sub_id': sub, 'ifold': ifold, 'set_id': str(set_id),'amplitude': amplitude}
                sub_cv_beta = sub_cv_beta.append(tmp_data, ignore_index=True)
    return sub_cv_beta

# set subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
data = data.query('game1_acc>=0.8')
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

stats_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/alignPhi/EC_park'
cmap = r'MspmT_0003.nii'
#roi_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/defROI/EC/EC_func_roi.nii'
roi_path = '/mnt/workdir/DCM/docs/Reference/Park_Grid_ROI/EC_Grid_roi.nii'
sub_cv_beta = extractBetas(subjects, roi_path, stats_dir, cmap)

savedir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/alignPhi/betas'
if not os.path.exists(savedir):
    os.mkdir(savedir)
save_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/alignPhi/betas/alignPhi_EC_park_ROI_betas.csv'
sub_cv_beta.to_csv(save_path,index=False)
#%%# set path template:

stats_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/M2_Decision/Setall/6fold/{}/spmF_0006.nii'
roi_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/M2_Decision/defROI/EC/check_EC_func_roi.nii'
savedir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/M2_Decision/betas'
if not os.path.exists(savedir):
    os.mkdir(savedir)
save_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/M2_Decision/betas/ROI-check_EC_betas_covariates.csv'

# set subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

folds = range(6,7)

sub_fold_beta = pd.DataFrame(columns=['sub_id','amplitude'])

for sub in subjects:
    stats_map = stats_path.format(sub)
    roi_img = load_img(roi_path)
    roi_img = resample_to_img(roi_img, stats_map,interpolation='nearest')
    amplitude = np.nanmean(apply_mask(imgs=stats_map, mask_img=roi_img))

    sub_tmp = sub.replace('-','_')
    age = data.loc[data.Participant_ID==sub_tmp,'Age'].values[0]
    acc = data.loc[data.Participant_ID==sub_tmp,'game1_acc'].values[0]
    tmp_data = {'sub_id': sub, 'amplitude': amplitude,'age':age,'acc':acc}
    sub_fold_beta = sub_fold_beta.append(tmp_data, ignore_index=True)
sub_fold_beta.to_csv(save_path,index=False)
#%%
import pandas as pd
import statsmodels.api as sm

X = sub_fold_beta[['age','acc']]
Y = sub_fold_beta['amplitude']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
model_summary = model.summary()
print(model_summary)

#%%
import plotly.express as px
fig = px.scatter_3d(sub_fold_beta, x='age', y='acc', z='amplitude')
fig.show()