#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:49:14 2022

@author: dell
"""
import os.path

import pandas as pd
from os.path import join as pjoin
import numpy as np
from scipy.stats import circmean,circstd
from nilearn.masking import apply_mask
from nilearn.image import math_img, resample_to_img, load_img


def estPhi(beta_sin_map, beta_cos_map, mask, ifold='6fold', level='roi'):
    ifold = int(ifold[0])

    if not np.array_equal(mask.affine, beta_sin_map.affine):
        mask = resample_to_img(mask, beta_sin_map, interpolation='nearest')
        print("Inconsistent affine matrix of two images.")

    beta_sin_roi = apply_mask(beta_sin_map, mask)
    beta_cos_roi = apply_mask(beta_cos_map, mask)

    if level == 'roi':
        mean_orientation = np.rad2deg(circmean(np.arctan2(beta_sin_roi, beta_cos_roi)) / ifold)
        std_orientation = np.rad2deg(circstd(np.arctan2(beta_sin_roi, beta_cos_roi)/ifold,np.pi/6,-np.pi/6))
        return mean_orientation,std_orientation
    elif level == 'voxel':
        mean_orientation = np.rad2deg(np.arctan2(beta_sin_roi, beta_cos_roi) / ifold)
        return mean_orientation
    else:
        raise Exception("The parameter-level is wrong.")


def estSubPhi(task,glm_type, sets, subjects, folds, savename, roi='group',level='roi'):
    # define input and output :
    dataroot = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/{}'.format(task, glm_type)
    datasink = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/{}/Phi'.format(task, glm_type)

    if not os.path.exists(datasink):
        os.mkdir(datasink)

    for set in sets:
        datadir = pjoin(dataroot, set)
        savepath = pjoin(datasink, savename)  # look out
        subs_phi = pd.DataFrame(columns=['sub_id', 'ifold', 'ec_phi', 'vmpfc_phi'])

        print("————————{} start!—————————".format(set))
        for sub in subjects:
            print(set, '-', sub)
            for ifold in folds:
                # load beta map
                bsin_path = pjoin(datadir, ifold, sub, 'Mcon_0010.nii')
                bcos_path = pjoin(datadir, ifold, sub, 'Mcon_0009.nii')
                beta_sin_map = load_img(bsin_path)
                beta_cos_map = load_img(bcos_path)

                # load roi
                if roi == 'group':
                    ec_roi = load_img('/mnt/workdir/DCM/docs/Reference/Park_Grid_ROI/EC_Grid_roi.nii')
                    vmpfc_roi = load_img('/mnt/workdir/DCM/docs/Reference/Park_Grid_ROI/mPFC_Grid_roi.nii')
                elif roi == 'individual':
                    ec_roi = load_img('/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/defROI/EC/individual'
                                  '/{}_EC_func_roi.nii'.format(sub.split('-')[-1]))
                    vmpfc_roi = load_img('/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/defROI/vmpfc/individual'
                                     '/{}_vmpfc_func_roi.nii'.format(sub.split('-')[-1]))
                else:
                    raise Exception("The is wrong.".format(roi))

                ec_phi = estPhi(beta_sin_map, beta_cos_map, ec_roi, ifold,level)
                vmpfc_phi = estPhi(beta_sin_map, beta_cos_map, vmpfc_roi, ifold,level)

                if isinstance(ec_phi,np.ndarray):
                    for i,(ep,vp) in enumerate(zip(ec_phi, vmpfc_phi)):
                        sub_phi = {'sub_id': sub, 'ifold': ifold,
                                   'ec_phi': ep, 'vmpfc_phi': vp,'voxel':i+1}
                        subs_phi = subs_phi.append(sub_phi, ignore_index=True)
                else:
                    ec_mean = ec_phi[0]
                    ec_std = ec_phi[1]

                    vmpfc_mean = vmpfc_phi[0]
                    vmpfc_std = vmpfc_phi[1]

                    sub_phi = {'sub_id': sub, 'ifold': ifold,
                               'ec_phi': ec_mean, 'ec_std':ec_std,
                               'vmpfc_phi': vmpfc_mean,'vmpfc_std':vmpfc_std}
                    subs_phi = subs_phi.append(sub_phi, ignore_index=True)
        subs_phi.to_csv(savepath, index=False)


if __name__  == "__main__":
    task = 'game1'
    glm_type = 'separate_hexagon'
    sets = ['Setall']
    folds = [str(i) + 'fold' for i in range(6, 7)]  # look out

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri==1')
    pid = data['Participant_ID'].to_list()
    subjects = [p.replace('_', '-') for p in pid]
    savename= 'estPhi_group_roi.csv'
    estSubPhi(task,glm_type,sets,subjects, folds, savename, roi='group',level='roi')
