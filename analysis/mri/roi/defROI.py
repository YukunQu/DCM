import os
import pandas as pd
from analysis.mri.roi.makeMask import makeSphereMask,makeSphereMask_coords
#%%
# Given coodinates, generate a ROI
stats_map = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group/all/2ndLevel/_contrast_id_ZF_0005/spmT_0001.nii'
coords = (49,92,42)
radius = (7,7,7)
savepath = r'/mnt/workdir/DCM/result/ROI/Group/mPFC_roi.nii.gz'
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

#%%
# Computing a Region of Interest (ROI) mask by nilearn
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn import image
from analysis.mri.img.zscore_nii import zscore_img
from nilearn.plotting import plot_stat_map

# load z-transfrom map
tmap = image.load_img(r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group/all/2ndLevel/_contrast_id_ZF_0005/spmT_0001.nii')

# get threshold map
img_data = tmap.get_fdata()
img_data[img_data<3.1] = 0
tmap_thr = image.new_img_like(tmap,img_data)
plot_stat_map(tmap_thr,title='', annotate=False,cut_coords=(0,0,0))
tmap_thr.to_filename('/mnt/workdir/DCM/result/ROI/Group/tmap_thr.nii.gz')

# self-computed mask
bin_tmap_thr = (img_data!=0)

# load peak point mask
peak_mask = image.get_data(image.load_img(r'/mnt/workdir/DCM/result/ROI/Group/mPFC_sphere_mask.nii.gz')).astype(bool)
bin_tmap_thr_peak_spere = np.logical_and(peak_mask, bin_tmap_thr)
#%%
from nilearn.plotting import plot_roi, show
bin_tmap_thr_peak_spere_img = image.new_img_like(tmap,bin_tmap_thr_peak_spere.astype(int))
plot_roi(bin_tmap_thr_peak_spere_img, cut_coords=(0,0,0))
bin_tmap_thr_peak_spere_img.to_filename("/mnt/workdir/DCM/result/ROI/Group/bin_tmap_thr_peak_spere_img.nii")
#%%
import numpy as np
from nilearn import image

roi = image.load_img(r'/mnt/workdir/DCM/result/ROI/Group/juelich_EC_R_prob-80.nii.gz')
roi_data = roi.get_fdata()
roi_data[roi_data<80]=0
bin_roi_data = (roi_data!=0)
new_roi = image.new_img_like(roi,bin_roi_data)
new_roi_resampled = image.resample_to_img(new_roi,r'/mnt/workdir/DCM/result/ROI/Group/tmap_thr.nii.gz','nearest')
new_roi_resampled.to_filename(r'/mnt/workdir/DCM/result/ROI/Group/juelich_EC_R_prob-80_new.nii.gz')