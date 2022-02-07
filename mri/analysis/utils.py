# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:03:30 2021

@author: QYK
"""

import numpy as np


from scipy.stats import circmean
from nilearn.masking import apply_mask
from nilearn.image import math_img,resample_to_img


def estimateMeanOrientation_Tim(beta_sin_map,beta_cos_map,mask,ifold=6):
    mask = resample_to_img(mask, beta_sin_map,interpolation='nearest')
    beta_sin_masked = apply_mask(beta_sin_map, mask)
    beta_sin = beta_sin_masked.mean()
    beta_cos_masked = apply_mask(beta_cos_map, mask)
    beta_cos = beta_cos_masked.mean()
    mean_oritation = np.rad2deg(np.arctan2(beta_sin,beta_cos))/ifold
    return mean_oritation 
        

def estimateMeanOrientation_Park(beta_sin_map,beta_cos_map,mask,ifold=6):
    fai_rad_map = math_img("np.arctan(img1/img2)",img1=beta_sin_map,img2=beta_cos_map)
    mask = resample_to_img(mask,fai_rad_map,interpolation='nearest')
    fai_rad_map_masked = apply_mask(fai_rad_map,mask)
    mean_oritation = np.rad2deg(circmean(fai_rad_map_masked)/ifold)
    return mean_oritation 

