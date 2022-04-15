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
from scipy.stats import circmean
from nilearn.masking import apply_mask
from nilearn.image import math_img,resample_to_img,load_img


def estPhi(beta_sin_map, beta_cos_map, mask, ifold='6fold',method='mean'):
    ifold = int(ifold[0])
    mask = resample_to_img(mask, beta_sin_map,interpolation='nearest')
    
    beta_sin_roi = apply_mask(beta_sin_map, mask)
    beta_cos_roi = apply_mask(beta_cos_map, mask)
    
    if method == 'mean':
        beta_sin = np.nanmean(beta_sin_roi)
        beta_cos = np.nanmean(beta_cos_roi)
        mean_orientation = np.rad2deg(np.arctan2(beta_sin,beta_cos))/ifold
        # mean_orientation = np.nanmean(np.rad2deg(np.arctan2(beta_sin_roi,beta_cos_roi)/ifold))
    elif method == 'circmean':
        mean_orientation = np.rad2deg(circmean(np.arctan(beta_sin_roi/beta_cos_roi))/ifold) # refer park's code
    else:
        print("Please specify the method.")
    return mean_orientation


if __name__ =="__main__":
    # define iterator
    training_sets = ['trainset1','trainset2']

    folds = [str(i)+'fold' for i in range(4,9)]

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query('(usable==1)&(game1_acc>0.75)&(Age>18)')
    pid = data['Participant_ID'].to_list()
    subjects = [p.replace('_','-') for p in pid]
    
    # define input and output :
    dataroot = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/training_set'
    datasink = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/Phi'

    if not os.path.exists(datasink):
        os.mkdir(datasink)
    
    for trainset in training_sets:
        datadir = pjoin(dataroot, trainset)
        save_path = pjoin(datasink,trainset+'_estPhi_individual_ROI.csv')
        subs_phi = pd.DataFrame(columns=['sub_id','ifold','ec_phi','vmpfc_phi'])
        
        print("————————{} start!—————————".format(trainset))
        for sub in subjects:
            print(trainset,'-',sub)
            for ifold in folds:
                # load beta map
                bsin_path = pjoin(datadir,ifold,sub,'Mcon_0002.nii')
                bcos_path = pjoin(datadir,ifold,sub,'Mcon_0001.nii')
                beta_sin_map = load_img(bsin_path)
                beta_cos_map = load_img(bcos_path)

                #     # load roi
                ec_roi = load_img(fr'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/defROI/EC/{sub}_EC_func_roi.nii')
                vmpfc_roi = load_img(fr'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/defROI/vmpfc/{sub}_vmpfc_func_roi.nii')
                #ec_roi = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/'
                #                  'osfstorage-archive/data/Analysis_ROI_nii/EC_Grid_roi.nii')
                #vmpfc_roi = load_img('/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/'
                #                     'osfstorage-archive/data/Analysis_ROI_nii/mPFC_Grid_roi.nii')

                ec_phi = estPhi(beta_sin_map, beta_cos_map, ec_roi,ifold,'mean')
                vmpfc_phi = estPhi(beta_sin_map, beta_cos_map, vmpfc_roi,ifold,'mean')
                
                sub_phi = {'sub_id':sub,'ifold':ifold,
                           'ec_phi':ec_phi,'vmpfc_phi':vmpfc_phi}
                subs_phi = subs_phi.append(sub_phi,ignore_index=True)
        subs_phi.to_csv(save_path,index=False)