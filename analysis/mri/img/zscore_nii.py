#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 00:03:50 2022

@author: dell
"""
import os
import numpy as np
import pandas as pd
from nilearn.image import load_img,math_img,new_img_like,get_data
from nilearn.masking import apply_mask


def zscore_img(filepath):
    mask = load_img(r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')
    #mask = load_img(r'/mnt/workdir/DCM/docs/Mask/res-02_desc-brain_mask.nii')
    fmap = load_img(filepath)

    fmap_mean = apply_mask(fmap, mask).mean()
    fmap_std = apply_mask(fmap, mask).std()
    fmap_zscore = math_img("(img- {})/{}".format(fmap_mean,fmap_std),img=fmap)

    fmap_data = get_data(fmap_zscore)
    mask_data = get_data(mask)
    fmap_data[mask_data==0] = np.float64('NaN')
    zscored_map = new_img_like(fmap_zscore, fmap_data)
    return zscored_map


def zscore_nii(source_dir,file,prefix):
    filepath = os.path.join(source_dir,file)
    mask = load_img(r'/mnt/workdir/DCM/docs/Mask/res-02_desc-brain_mask.nii')
    fmap = load_img(filepath)

    fmap_mean = apply_mask(fmap, mask).mean()
    fmap_std = apply_mask(fmap, mask).std()
    fmap_zscore = math_img("(img- {})/{}".format(fmap_mean,fmap_std),img=fmap)

    fmap_data = get_data(fmap_zscore)
    mask_data = get_data(mask)
    fmap_data[mask_data == 0] = np.float64('NaN')
    map_zscore = new_img_like(fmap_zscore, fmap_data)
    map_zscore.to_filename(os.path.join(source_dir,prefix+file[3:]))


#%%
fmap = r'/mnt/workdir/DCM/result/tmp_test/average_fmap_across_sub/spmF_0005.nii'
zscore_map = zscore_img(fmap)
zscore_map.to_filename('/mnt/workdir/DCM/result/tmp_test/average_fmap_across_sub/ZF_0005.nii')

#%%
if __name__ == "__main__":
    # zscore the 1st level result
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()

    for ifold in range(8,9):
        cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/decision_grid_rsa/Setall/6fold/{}/rs-corr_img_coarse_{}fold.nii'
        save_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/decision_grid_rsa/Setall/6fold/{}/rs-corr_zmap_coarse_{}fold.nii'
        for sub_id in subjects:
            zscored_map = zscore_img(cmap_template.format(sub_id,ifold))
            zscored_map.to_filename(save_template.format(sub_id,ifold))
            print("The map of {} have been zscored.".format(sub_id))
        print("{}fold have been completed.".format(ifold))