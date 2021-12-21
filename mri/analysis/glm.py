# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:04:57 2021

@author: QYK
"""

import numpy as np
from nilearn.glm.first_level import FirstLevelModel


def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def glm1_estimate_fai(func_img,design_matrices):
    # fit  glm to estimate mean orientation
    time_ref = 1.46/3
    fmri_glm = FirstLevelModel(smoothing_fwhm=8,slice_time_ref=time_ref)
    fmri_glm = fmri_glm.fit(func_img, design_matrices=design_matrices) 
    
    # define contrast
    contrast_matrix = np.eye(design_matrices[0].shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                    for i, column in enumerate(design_matrices[0].columns)])
    
    hexagon_modulation = np.vstack([basic_contrasts['sin'],
                                    basic_contrasts['cos']])
    
    # F-test
    Fmap = fmri_glm.compute_contrast(hexagon_modulation,stat_type='F',output_type='z_score')

    # beta_cos and beta_sin map
    beta_sin_map = fmri_glm.compute_contrast(basic_contrasts['sin'], output_type='effect_size')
    beta_cos_map = fmri_glm.compute_contrast(basic_contrasts['cos'], output_type='effect_size')
    return Fmap, beta_sin_map, beta_cos_map


def glm2_estimate_hexagon_effect(func_img,design_matrices):
    # fit glm to estimate hexagon moudlation effect
    time_ref = 1.46/3
    fmri_glm = FirstLevelModel(smoothing_fwhm=8,slice_time_ref=time_ref)
    fmri_glm = fmri_glm.fit(func_img, design_matrices=design_matrices) 
    
    # define contrast
    contrast_matrix = np.eye(design_matrices[0].shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                    for i, column in enumerate(design_matrices[0].columns)])
    hexagon_modulation = basic_contrasts['fai_modulation']
    
    # map: beta(cos(theta-fai))
    t_map = fmri_glm.compute_contrast(hexagon_modulation,output_type='z_score')
    return t_map


def glm3_alignVSmisalign(func_img,design_matrices,mask,save_path):
    # fit glm2 to see the contrast between align and misalign condition
    pass 

