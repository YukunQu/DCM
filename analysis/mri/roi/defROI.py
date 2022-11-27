import os
import pandas as pd
from analysis.mri.roi.makeMask import makeSphereMask,makeSphereMask_coords
#%%
# Given coodinates, generate a ROI
stats_map = r'/mnt/data/DCM/result_backup/2022.11.22/game1/separate_hexagon_2phases_correct_trials/Setall/group/all/2ndLevel/_contrast_id_ZF_0005/spmT_0001.nii'
coords = (48,37,67)
radius = (5/3,5/3,5/3)
savepath = r'/mnt/workdir/DCM/result/ROI/Group/PCC_roi.nii.gz'
makeSphereMask_coords(stats_map,savepath,coords,radius)

#%%
# Generate group ROI
stats_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/separate_hexagon_correct_trials_train/Set2/group/all/ 2ndLevel/_contrast_id_ZF_0005/spmT_0001.nii'
mask = r'/mnt/workdir/DCM/docs/Mask/VMPFC/VMPFC_nilearn.nii.gz'
savepath = r'/mnt/workdir/DCM/result/ROI/Group/mPFC_m2_set2_roi.nii.gz'
radius = (5/3,5/3,5/3)
makeSphereMask(stats_map, mask, savepath, radius=radius)
#%%
# Generate individual ROI

# get subjects list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'game1_fmri==1')  # look out
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]

# get all path of stats_map
stats_map_template = r''
savepath_template = r''
mask = r''
radius = (5/3,5/3,5/3)

for sub in subject_list:
    stats_map = stats_map_template.format(sub)
    savepath = savepath_template.format(sub)
    roi = r'/mnt/workdir/DCM/docs/Reference/EC_ROI/volume/EC-thr50-2mm.nii.gz'
    makeSphereMask(stats_map, roi, savepath, radius=radius)