import os
import pandas as pd
from analysis.mri.roi.makeMask import makeSphereMask, makeSphereMask_coords


def defROI_by_coord():
    # Given coodinates, generate a ROI
    stats_map = r'/mnt/workdir/DCM/docs/Reference/Alexa/data/fig2MNI152NL.nii.gz'
    coords = (49, 85, 43)
    radius = (2.5, 2.5, 2.5)
    savepath = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/group_nilearn/age/mPFC_Peak_ROI.nii.gz'
    makeSphereMask_coords(stats_map, savepath, coords, radius)


def defROI_by_group_peak():
    """Generate Group ROI accoording to the peak point in mask"""
    stats_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/separate_hexagon_correct_trials_train/Set2/group/all/ 2ndLevel/_contrast_id_ZF_0005/spmT_0001.nii'
    mask = r'/mnt/workdir/DCM/docs/Mask/VMPFC/VMPFC_nilearn.nii.gz'
    savepath = r'/mnt/workdir/DCM/result/ROI/Group/mPFC_m2_set2_roi.nii.gz'
    radius = (5 / 3, 5 / 3, 5 / 3)
    makeSphereMask(stats_map, mask, savepath, radius=radius)


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
from analysis.mri.img.zscore_nii import zscore_img
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_roi

# load z-transfromed map
tmap = image.load_img(r'/mnt/data/DCM/result_backup/2023.1.2/game1/grid_rsa_8mm/Setall/6fold/group/hp-adult'
                      r'/2ndLevel/_contrast_id_rs-corr_zmap_coarse/spmT_0001.nii')

# get threshold map
img_data = tmap.get_fdata()
img_data[img_data < 4] = 0
tmap_thr = image.new_img_like(tmap, img_data)
plot_stat_map(tmap_thr, title='', annotate=False, cut_coords=(0, 0, 0))
bin_tmap_thr = (img_data != 0)  # self-computed mask

# load peak point mask/ anatomaical mask
mask = image.get_data(image.load_img(r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL.nii.gz'))
bin_tmap_thr_masked = np.logical_and(mask.astype(bool), bin_tmap_thr)
# plot roi and save
bin_tmap_thr_peak_spere_img = image.new_img_like(tmap, bin_tmap_thr_masked.astype(int))
plot_roi(bin_tmap_thr_peak_spere_img, cut_coords=(0, 0, 0))
bin_tmap_thr_peak_spere_img.to_filename("/mnt/workdir/DCM/result/ROI/Group/RSA-EC_thr4.nii.gz")
