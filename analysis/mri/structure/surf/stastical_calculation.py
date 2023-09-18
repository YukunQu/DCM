import os

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

#%%
metrics = pd.read_csv(r'/mnt/workdir/DCM/Result/analysis/brain_metrics.csv')

# dti_metrics = metrics.dropna()
# pearsonr(dti_metrics['Grid-like code (Game1)'], dti_metrics['EC.MD'])

metrics['mPFC.thickness'] = (metrics['lh.mPFC.thickness'] + metrics['rh.mPFC.thickness'])/2
metrics['mPFC.volume'] = (metrics['lh.mPFC.volume'] + metrics['rh.mPFC.volume'])/2

# # Initialize the StandardScaler
# scaler = StandardScaler()
# # Iterate through each column and apply z-score scaling only to non-NaN values
# for column in metrics.columns:
#     # Get the non-NaN values in the current column
#     non_nan_values = metrics[column][~np.isnan(metrics[column])]
#
#     # Apply z-score scaling to the non-NaN values
#     scaled_values = scaler.fit_transform(non_nan_values.to_numpy().reshape(-1, 1))
#
#     # Update the original DataFrame with the scaled values
#     metrics.loc[~np.isnan(metrics[column]), column] = scaled_values.ravel()

#%% multi-regression
# using GLM to predict inference accuracy from brain activity and age
metrics = metrics.dropna(subset='mPFC.FA')
X = metrics[['mPFC.MD','Distance code (Game2)']]
Y = metrics['game2_test_acc']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
model_summary = model.summary()
print(model_summary)

#%% mediation
iv = 'EC.MD'
m = ['Age']
dv = 'Grid-like code (Game1)'
cov = ['Distance code (Game1)']
scaler = StandardScaler()
metrics[['EC.MD','Age','Grid-like code (Game1)']] = scaler.fit_transform(metrics[['EC.MD','Age','Grid-like code (Game1)']])

# Mediation analysis
mediation_results,indirect_beta_sample = pg.mediation_analysis(data=metrics, x=iv, m=m, y=dv,seed=42,n_boot=10000,return_dist=True)
#add titile for results:
print(f'Mediation analysis for {iv} on {dv} \n'
      f'with {m} by contorlling {cov}')
print(mediation_results.round(3))

#%%
hemis = ['lh','rh']
measures = ['thickness','volume']
strucutral_measures = [f'aparcstats2table_{hemi}_{mea}.txt' for hemi in hemis for mea in measures]
strucutral_file_dir = '/mnt/workdir/DCM/Result/analysis/structure'

beh_measures = ['Age','game1_acc']

metrics = pd.read_csv(r'/mnt/workdir/DCM/Result/analysis/brain_metrics.csv')

for bm in beh_measures:
    corr_dict = {}
    for sm in strucutral_measures:
        sm_file = pd.read_table(os.path.join(strucutral_file_dir, sm))
        sm_sub_list = sm_file.iloc[:,0].to_list()

        beh_sub_list = metrics['Participant_ID'].to_list()

        if sm_sub_list != beh_sub_list:
            raise Exception("The subjects in two measures talbe is not matched.")

        parc_rois = sm_file.columns[1:]
        sm_corr = {}
        for pr in parc_rois:
            r, p = pearsonr(sm_file[pr],metrics[bm])
            if 'Mean' in pr:
                continue
            else:
                pr = pr.replace("_volume",'')
                pr = pr.replace("_thickness",'')
                sm_corr[pr] = r
        parts = sm.split('_')
        sm_name = parts[1] + '.' + parts[2].split('.')[0]
        corr_dict[sm_name] = sm_corr

    corr_df = pd.DataFrame(corr_dict)
    corr_df.to_csv(rf'/mnt/workdir/DCM/Result/analysis/structure/measures_corr-{bm}.csv')

#%%
hemis = ['lh','rh']
measures = ['FA','MD']
strucutral_measures = [f'aparc{hemi}_{mea}.txt' for hemi in hemis for mea in measures]
strucutral_file_dir = '/mnt/workdir/DCM/Result/analysis/structure'

beh_measures = ['Age','game1_acc']

metrics = pd.read_csv(r'/mnt/workdir/DCM/Result/analysis/brain_metrics.csv')

for bm in beh_measures:
    corr_dict = {}
    for sm in strucutral_measures:
        sm_file = pd.read_table(os.path.join(strucutral_file_dir, sm))
        sm_sub_list = sm_file.iloc[:,0].to_list()

        beh_sub_list = metrics['Participant_ID'].to_list()

        if sm_sub_list != beh_sub_list:
            raise Exception("The subjects in two measures talbe is not matched.")

        parc_rois = sm_file.columns[1:]
        sm_corr = {}
        for pr in parc_rois:
            r, p = pearsonr(sm_file[pr],metrics[bm])
            if 'Mean' in pr:
                continue
            else:
                pr = pr.replace("_volume",'')
                pr = pr.replace("_thickness",'')
                sm_corr[pr] = r
        parts = sm.split('_')
        sm_name = parts[1] + '.' + parts[2].split('.')[0]
        corr_dict[sm_name] = sm_corr

    corr_df = pd.DataFrame(corr_dict)
    corr_df.to_csv(rf'/mnt/workdir/DCM/Result/analysis/structure/measures_corr-{bm}.csv')