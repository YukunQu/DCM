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
from nilearn.image import math_img, resample_to_img, load_img


def estPhi(beta_sin_map, beta_cos_map, mask, ifold='6fold', method='weighted average'):
    ifold = int(ifold[0])
    if not np.array_equal(mask.affine, beta_sin_map.affine):
        print("Inconsistent affine matrix of two images.")
        mask = resample_to_img(mask, beta_sin_map, interpolation='nearest')

    beta_sin_roi = apply_mask(beta_sin_map, mask)
    beta_cos_roi = apply_mask(beta_cos_map, mask)

    if method == 'circmean':
        mean_orientation = np.rad2deg(circmean(np.arctan2(beta_sin_roi, beta_cos_roi))/ifold)
        return mean_orientation
    elif method == 'mean':
        orientation = np.rad2deg(np.arctan2(beta_sin_roi.mean(), beta_cos_roi.mean()))
        if orientation < 0:
            orientation += 360
        mean_orientation = orientation/ifold
        return mean_orientation
    elif method == 'weighted average':
        # calculate weight
        amplitude = np.sqrt(beta_sin_roi**2 + beta_cos_roi**2)
        weight = amplitude/np.sum(amplitude)
        # calculate orientations in each voxel
        population_vector = np.rad2deg(np.arctan2(beta_sin_roi, beta_cos_roi))
        population_vector = [o+360 if o < 0 else o for o in population_vector]
        population_vector = np.array(population_vector)/ifold
        # weighted average
        mean_orientation = np.sum(population_vector * weight)
        return mean_orientation
    elif method == 'orientation vector':
        angle_vector = np.rad2deg(np.arctan2(beta_sin_roi, beta_cos_roi))
        orientation_vector = []
        for angle in angle_vector:
            if angle<0:
                angle+=360
            orientation_vector.append(angle/ifold)
        return orientation_vector
    else:
        raise Exception("The specify method is wrong.")


if __name__ == "__main__":
    # subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')
    data = data.query("(game1_acc>=0.80)and(Age>=18)")
    subjects = data['Participant_ID'].to_list()

    # set sin_cmap and cos_cmap
    cos_cmap_template = '/mnt/data/DCM/result_backup/2023.1.2/game1/cv_train1/Setall/{}/{}/con_0001.nii'
    sin_cmap_template = '/mnt/data/DCM/result_backup/2023.1.2/game1/cv_train1/Setall/{}/{}/con_0002.nii'

    # set ROI
    roi = r'/mnt/workdir/DCM/result/ROI/Group/RSA-EC_thr3.1.nii.gz'

    # set output
    outdir = r'/mnt/workdir/DCM/result/CV/Phi/2023.1.2'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    savepath = os.path.join(outdir,'estPhi_ROI-RSA-ECthr3.1_On-M2_trials-odd_subjects-hp_orientation_vector.csv')

    folds = [str(i)+'fold' for i in range(6,7)]
    subs_phi = pd.DataFrame(columns=['sub_id', 'ifold', 'Phi'])

    for sub in subjects:
        for ifold in folds:
            # load cmap
            bcos_path = cos_cmap_template.format(ifold,sub)
            bsin_path = sin_cmap_template.format(ifold,sub)

            cos_cmap = load_img(bcos_path)
            sin_cmap = load_img(bsin_path)
            # load roi
            mask = load_img(roi)
            # extract Phi
            method = 'orientation vector'
            if method == 'orientation vector':
                phis = estPhi(sin_cmap, cos_cmap, mask,ifold=ifold,method=method)
                for i,phi in enumerate(phis):
                    sub_phi = {'sub_id': sub, 'ifold': ifold,'Phi': phi,'voxel_index':i+1}
                    subs_phi = subs_phi.append(sub_phi, ignore_index=True)
            else:
                phi = estPhi(sin_cmap, cos_cmap, mask,ifold=ifold,method=method)
                sub_phi = {'sub_id': sub, 'ifold': ifold,'Phi': phi}
                subs_phi = subs_phi.append(sub_phi, ignore_index=True)
    subs_phi.to_csv(savepath, index=False)