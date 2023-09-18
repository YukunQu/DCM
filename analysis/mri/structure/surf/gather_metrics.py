import os
import numpy as np
import pandas as pd


def read_aseg(subjects_list,smeasure='Right-Hippocampus'):
    fs_dir = "/mnt/workdir/DCM/BIDS/derivatives/freesurfer"
    measures = []
    for sid in subjects_list:
        # Define the path to the aseg.stats file
        stats_file = os.path.join(fs_dir, sid, "stats", "aseg.stats")

        # Open the stats file and read its contents
        with open(stats_file, "r") as f:
            stats_data = f.readlines()

        # Find the line that contains the Total cortical gray matter volume information
        for line in stats_data:
            if smeasure in line:
                # Extract the volume value from the line
                measure = float(line.split()[3])
                measures.append(measure)
                break
    return measures
#%%
# gather the metrics into one files
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()

# load SBM metrics
hemis = ['lh','rh']
rois = ['EC','mPFC']
stats_names = ['thickness','volume']

new_data = data[['Participant_ID','game1_acc','game2_test_acc','game1_fmri','game2_fmri','Sex ']].copy()

for stats_name in stats_names:
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

            column_name = f'{hemi}.{roi}.{stats_name}'
            new_data[column_name] = stats

#%%
# load DTI metrics
dti_metrics_name = ['FA','MD']
dti_metric_file = '/mnt/workdir/DCM/Result/analysis/structure/MRtrix/metrics/DMN_dti_metrics_AllData.csv'

val_sub_info = participants_data.query("game1_fmri>=0.5")
val_sub = val_sub_info['Participant_ID'].to_list()

# load vbm file
dti_metric = pd.read_csv(dti_metric_file)
sub_id_list = dti_metric['sub-id'].to_list()

# compare sub_id_list and val_sub
d1 = []
d2 = []
for s in val_sub:
    if s not in sub_id_list: d1.append(s)
print(":Those validate subjects not have diffusion files:", d1)
for s in sub_id_list:
    if s not in val_sub:d2.append(s)
print(":The subjects with diffusion files are not validated:", d2)
diff = d1+d2

# Remove the rows from val_sub_info whose Participant_ID is present in the diff list.
val_sub_info = val_sub_info[~val_sub_info['Participant_ID'].isin(diff)].sort_values('Participant_ID')
dti_metric = dti_metric[~dti_metric['sub-id'].isin(diff)].sort_values('sub-id')

val_sub = val_sub_info['Participant_ID'].to_list()
sub_id_list = dti_metric['sub-id'].to_list()

# recheck if the val_sub equal to sub_id_list
if val_sub == sub_id_list:
    print("val_sub equals sub_id_list")
else:
    print("val_sub does not equal sub_id_list")

roi_names = dti_metric['roi'].unique().tolist()
dti_metric_names = dti_metric['metrics'].unique().tolist()

for index, row in dti_metric.iterrows():
    sub_id = row['sub-id']
    roi = row['roi']
    dti_metric_name = row['metrics']
    value = row['value']
    new_data.loc[new_data['Participant_ID'] == sub_id, f'{roi}.{dti_metric_name}'] = value

#%%
# load hippocampus metrics (volume and FA)
# load volume
rhc = read_aseg(subject_list,'Right-Hippocampus')
lhc = read_aseg(subject_list,'Left-Hippocampus')
new_data['rHC.volume'] = rhc
new_data['lHC.volume'] = lhc
new_data['HC.volume'] = new_data['rHC.volume'] + new_data['lHC.volume']

#%%
from nilearn import image
from nilearn import masking

# load cognitive map representation
func_templates = {'Distance code (Game1)':r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/'
                                          r'distance_spct/Setall/6fold/{}/zmap/distance_zmap.nii.gz',
                  'Distance code (Game2)':r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/'
                                          r'distance_spct/Setall/6fold/{}/zmap/distance_zmap.nii.gz',
                  'Grid-like code (Game1)':r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/'
                                           r'grid_rsa_corr_trials/Setall/6fold/{}/rsa/rsa_zscore_img_coarse_6fold.nii.gz',
                  'Map-alignment (Game2)':r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2'
                                          r'/cv_hexagon_spct/Setall/6fold/{}/cmap/alignPhi_cmap.nii.gz'}

for subjid in subject_list:
    for rep,cmap_tmp in func_templates.items():
        func_template = func_templates[rep]
        func_path = func_template.format(subjid)
        # check the existence of the functional file
        if not os.path.exists(func_path):
            # print(subjid, "doesn't exist.")
            new_data.loc[new_data['Participant_ID'] == sub_id, rep] = np.nan
        else:
            func_data = image.load_img(func_path)
            if 'Distance' in rep:
                mask = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/VMPFC/VMPFC_MNI152NL_new.nii.gz')
            else:
                mask = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz')
                mask = image.binarize_img(mask,5)
            func_data = masking.apply_mask(func_data,mask)
            new_data.loc[new_data['Participant_ID'] == subjid, rep] = np.mean(func_data)

#%%
new_data['Age'] = data['Age']
new_data.to_csv(r'/mnt/workdir/DCM/Result/analysis/brain_metrics_game1_20230914.csv', index=False)