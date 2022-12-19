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


def estPhi(beta_sin_map, beta_cos_map, mask, ifold='6fold', method='mean'):
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
        mean_orientation = np.rad2deg(np.arctan2(beta_sin_roi.mean(), beta_cos_roi.mean())/ifold)
        return mean_orientation
    else:
        raise Exception("The specify method is wrong.")


if __name__ == "__main__":
    # subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game2_fmri>0.5')
    subjects = data['Participant_ID'].to_list()

    # set sin_cmap and cos_cmap
    cos_cmap_template = '/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/{}/{}/con_0001.nii'
    sin_cmap_template = '/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/{}/{}/con_0002.nii'

    # set ROI
    roi = r'/mnt/workdir/DCM/result/ROI/Group/F-test_mPFC_thr2.3.nii.gz'

    # set output
    outdir = r'/mnt/workdir/DCM/result/CV/Phi/2022.12.21'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    savepath = os.path.join(outdir,'estPhi_ROI-bigmPFC_On-M2_trial-corr_subject-all.csv')

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
            phi = estPhi(sin_cmap, cos_cmap, mask,ifold=ifold,method='circmean')
            sub_phi = {'sub_id': sub, 'ifold': ifold,'Phi': phi}
            subs_phi = subs_phi.append(sub_phi, ignore_index=True)
    subs_phi.to_csv(savepath, index=False)