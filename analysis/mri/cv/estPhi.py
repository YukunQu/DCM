#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: dell
"""
import os.path

import pandas as pd
from os.path import join as pjoin
import numpy as np
from scipy.stats import circmean,circstd
from nilearn.masking import apply_mask
from nilearn.image import math_img, resample_to_img, load_img,binarize_img


def estPhi(sin_beta_map, cos_beta_map, mask, ifold='6fold', method='circmean'):
    ifold = int(ifold[0])
    if not np.array_equal(mask.affine, sin_beta_map.affine):
        print("Inconsistent affine matrix for two images.")
        mask = resample_to_img(mask, sin_beta_map, interpolation='nearest')

    sin_betas = apply_mask(sin_beta_map, mask)
    cos_betas = apply_mask(cos_beta_map, mask)

    if method == 'circmean':
        orientation = np.arctan2(sin_betas, cos_betas)
        mean_orientation = np.rad2deg(circmean(orientation)/ifold)
        std_orientation = np.rad2deg(circstd(orientation)/ifold)
    elif method == 'mean':
        mean_orientation = np.rad2deg(np.arctan2(sin_betas.mean(), cos_betas.mean()))
        std_orientation = np.std(np.rad2deg(np.arctan2(sin_betas, cos_betas))/ifold)
        if mean_orientation < 0:
            mean_orientation += 360
        mean_orientation = mean_orientation/ifold
    elif method == 'wmean':
        # weighted average of angles for different voxels
        # calculate weight
        amplitude = sin_betas**2 + cos_betas**2
        weight = amplitude/np.sum(amplitude)
        # calculate orientations in each voxel
        population_vector = np.rad2deg(np.arctan2(sin_betas, cos_betas))
        population_vector = [o+360 if o < 0 else o for o in population_vector]
        population_vector = np.array(population_vector)/ifold
        # weighted average
        mean_orientation = np.sum(population_vector * weight)
        std_orientation = np.std(population_vector * weight)  # weighted average should
    else:
        raise Exception("The specify method is wrong.")
    return mean_orientation,std_orientation


def estimate_game1_cv_phi(workdir):
    # set fold
    folds = [str(i)+'fold' for i in range(6,7)]

    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')
    subjects = data['Participant_ID'].to_list()

    # set sin_cmap and cos_cmap for odd trials and even trials
    # odd trials
    cos_cmap_template = pjoin(workdir,'Setall/{}/{}/cmap/cos_{}_cmap.nii.gz')
    sin_cmap_template = pjoin(workdir,'Setall/{}/{}/cmap/sin_{}_cmap.nii.gz')

    # set ROI
    from nilearn import image
    roi = pjoin(workdir,'EC_thr3.1.nii.gz')
    #roi1 = image.load_img(r'/mnt/data/DCM/tmp/aparc/mask/lh.entorhinal.nii.gz')
    #roi2 = image.load_img(r'/mnt/data/DCM/tmp/aparc/mask/rh.entorhinal.nii.gz')
    #roi = image.math_img('img1+img2',img1=roi1,img2=roi2)
    #roi = r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz'
    mask = load_img(roi)
    #mask = binarize_img(mask,10)

    # set method
    method = 'circmean'

    # set output
    savepath = os.path.join(workdir,f'estPhi_ROI-EC-cvtrian_{method}_cv.csv')

    subs_phi = pd.DataFrame(columns=['ifold','sub_id','trial_type','Phi_mean','Phi_std'])

    for ifold in folds:
        for sub in subjects:
            for trial_type in ['odd','even']:
                # load cmap
                bcos_path = cos_cmap_template.format(ifold,sub,trial_type)
                bsin_path = sin_cmap_template.format(ifold,sub,trial_type)

                cos_cmap = load_img(bcos_path)
                sin_cmap = load_img(bsin_path)

                # calculate mean orientation and std of orientaion
                mphi,sphi = estPhi(sin_cmap, cos_cmap,mask,ifold=ifold,method=method)
                subs_phi = subs_phi.append({'ifold': ifold,'sub_id':sub,'trial_type':trial_type,
                                            'Phi_mean': mphi,'Phi_std':sphi}, ignore_index=True)
    subs_phi.to_csv(savepath, index=False)


def estimate_game1_whole_trials_phi(workdir):
    # set fold
    folds = [str(i)+'fold' for i in range(6,7)]

    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')
    subjects = data['Participant_ID'].to_list()

    # set sin_cmap and cos_cmap for odd trials and even trials
    # odd trials
    cos_cmap_template = pjoin(workdir,'Setall/{}/{}/cmap/cos_cmap.nii.gz')
    sin_cmap_template = pjoin(workdir,'Setall/{}/{}/cmap/sin_cmap.nii.gz')

    # set ROI
    roi = pjoin(workdir,'EC_thr3.1.nii.gz')
    mask = load_img(roi)

    # set method
    method = 'circmean'

    # set output
    savepath = os.path.join(workdir,f'estPhi_ROI-EC_{method}_trial-all.csv')

    subs_phi = pd.DataFrame(columns=['ifold','sub_id','trial_type','Phi_mean','Phi_std'])

    for ifold in folds:
        for sub in subjects:
            for trial_type in ['all']:
                # load cmap
                bcos_path = cos_cmap_template.format(ifold,sub)
                bsin_path = sin_cmap_template.format(ifold,sub)

                cos_cmap = load_img(bcos_path)
                sin_cmap = load_img(bsin_path)

                # calculate mean orientation and std of orientaion
                mphi,sphi = estPhi(sin_cmap, cos_cmap,mask,ifold=ifold,method=method)
                subs_phi = subs_phi.append({'ifold': ifold,'sub_id':sub,'trial_type':trial_type,
                                            'Phi_mean': mphi,'Phi_std':sphi}, ignore_index=True)
    subs_phi.to_csv(savepath, index=False)


if __name__ == "__main__":
    workingdir = r'/mnt/data/DCM/result_backup/2023.5.17/cv_train_hexagon_spct'
    estimate_game1_cv_phi(workingdir)
    #estimate_game1_whole_trials_phi(workingdir)