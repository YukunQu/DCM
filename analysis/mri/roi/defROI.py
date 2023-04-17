import os
from nilearn.image import binarize_img
import pandas as pd
from analysis.mri.roi.makeMask import makeSphereMask, makeSphereMask_coords


def defROI_by_coord():
    # Given coodinates, generate a ROI
    stats_map  =r'/mnt/workdir/DCM/docs/Mask/VMPFC/VMPFC_nilearn.nii.gz'
    coords = (48,86, 45)
    radius = (2.5, 2.5, 2.5)
    savepath = r'/mnt/data/DCM/result_backup/2023.3.24/Nilearn_smodel/game1/distance_spct/Setall/ACC_ROI.nii.gz'
    makeSphereMask_coords(stats_map, savepath, coords, radius)


def defROI_by_group_thr():
    """Generate Group ROI accoording to the peak point in mask"""
    stats_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_separate_phases_correct_trials/Setall/6fold/group_sm8/mean/hexagon_tmap.nii.gz'
    mask = r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL_prob.nii.gz'
    mask_bin = binarize_img(mask)
    savepath = r'/mnt/workdir/DCM/result/ROI/Group_nilearn/hexagon_EC_ROI.nii.gz'
    radius = (2.5, 2.5, 2.5)
    makeSphereMask(stats_map, mask_bin, savepath, radius=radius)


def defROI_by_group_peak():
    """Generate Group ROI accoording to the peak point in mask"""
    stats_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_separate_phases_correct_trials/Setall/6fold/group_sm8/mean/hexagon_tmap.nii.gz'
    mask = r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL_prob.nii.gz'
    mask_bin = binarize_img(mask)
    savepath = r'/mnt/workdir/DCM/result/ROI/Group_nilearn/hexagon_EC_ROI.nii.gz'
    radius = (2.5, 2.5, 2.5)
    makeSphereMask(stats_map, mask_bin, savepath, radius=radius)

def defROI_by_indivi_peak():
    """Generate individual ROI accoording to the peak point in mask"""
    # get subjects list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'game1_fmri>=0.5')  # look out
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('_')[-1] for p in pid]

    # get all path of stats_map
    stats_map_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/m2_hexagon_correct_trials/Setall/6fold/{}/ZF_0005.nii'
    savepath_template = r'/mnt/workdir/DCM/result/ROI/individual/{}_roi.nii.gz'
    mask = r'/mnt/workdir/DCM/docs/Mask/VMPFC/VMPFC_nilearn.nii.gz'
    radius = (2, 2, 2)
    for sub in subject_list:
        stats_map = stats_map_template.format(sub)
        savepath = savepath_template.format(sub)
        makeSphereMask(stats_map, mask, savepath, radius=radius)


def threshold_binary_img(source_img, thr):
    from nilearn import image
    roi = image.load_img(source_img)
    roi_data = roi.get_fdata()
    roi_data[roi_data < thr] = 0
    bin_roi_data = (roi_data != 0)
    new_roi = image.new_img_like(roi, bin_roi_data)
    return new_roi


# %%
# Computing a Region of Interest (ROI) mask by nilearn
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn import image
#from analysis.mri.img.zscore_nii import zscore_img
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_roi

# load statistical map and mask
stats_map = image.load_img(r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_distance_spct/'
                           r'Setall/6fold/group_203/mean/hexagon_zmap.nii.gz')
mask = image.load_img(r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL_prob.nii.gz')
if not np.array_equal(mask.affine,stats_map.affine):
    raise Exception("The mask and statistical map have different affine matrix.")

# get threshold map of statistical map
img_data = stats_map.get_fdata()
img_data[img_data <= 3.1] = 0
stats_map_thr = image.new_img_like(stats_map, img_data)
plot_stat_map(stats_map_thr, title='', annotate=False, cut_coords=(0, -4, 0))
bin_tmap_thr = (img_data != 0)

# mask the thresholded statistical map
mask_data = image.get_data(mask)
mask_data[mask_data<10] = 0
mask_thr = image.new_img_like(mask,mask_data)
plot_stat_map(mask_thr, title='', annotate=False, cut_coords=(0, -4, 0))

bin_tmap_thr_masked = np.logical_and(mask_data.astype(bool), bin_tmap_thr)

# plot roi and save
bin_tmap_thr_peak_spere_img = image.new_img_like(stats_map, bin_tmap_thr_masked.astype(int))
plot_roi(bin_tmap_thr_peak_spere_img, cut_coords=(0, 0, 0))
bin_tmap_thr_peak_spere_img.to_filename("/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_distance_spct/EC_thr3.1.nii.gz")
