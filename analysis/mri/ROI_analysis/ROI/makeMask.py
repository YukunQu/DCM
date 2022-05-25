#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:54:13 2022

@author: dell
"""
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img,resample_to_img,binarize_img,new_img_like
from nltools.mask import create_sphere
from nltools.data import Brain_Data
from nibabel.affines import apply_affine


def get_coordinate(image, mask):
    # return voxel coordinates of peak point
    mask = resample_to_img(mask,image,interpolation='nearest')
    img_masked = apply_mask(image, mask)
    peak_value = np.max(img_masked)
    print('Peak value:',peak_value)    
    data = image.get_fdata()
    coords = np.where(data==peak_value)
    peak_coords = [coords[0][0], coords[1][0], coords[2][0]]
    return peak_coords
    

def sphere_roi(voxloc, radius, value, datashape = (91,109,91), data = None):
    """
    Generate a sphere roi which centered in (x,y,z)
    Parameters:
        voxloc: (x,y,z), center vox of spheres
        radius: radius (unit: vox), note that it's a list
        value: label value 
        datashape: data shape, by default is (91,109,91)
        data: Add sphere roi into your data, by default data is an empty array
    output:
        data: sphere roi image data
        loc: sphere roi coordinates
    """
    if data is not None:
        try:
            if data.shape != datashape:
                raise Exception('Data shape is not consistent with parameter datashape')
        except AttributeError:
            raise Exception('data type should be np.array')
    else:
        data = np.zeros(datashape)

    loc = []
    for n_x in range(int(voxloc[0]-radius[0]), int(voxloc[0]+radius[0]+1)):
        for n_y in range(int(voxloc[1]-radius[1]), int(voxloc[1]+radius[1]+1)):
            for n_z in range(int(voxloc[2]-radius[2]), int(voxloc[2]+radius[2]+1)):
                n_coord = np.array((n_x, n_y, n_z))
                coord = np.array((voxloc[0], voxloc[1], voxloc[2]))
                minus = coord - n_coord
                if (np.square(minus) / np.square(np.array(radius)).astype(np.float32())).sum()<=1:
                    try:
                        data[n_x, n_y, n_z] = value
                        loc.append([n_x,n_y,n_z])
                    except IndexError:
                        pass
    loc = np.array(loc)
    return data, loc


def makeSphereMask(imgpath,mask_path,savepath,radius=(2,2,2),label=1,coords=None):
    image = load_img(imgpath)
    mask = load_img(mask_path)
    mask = resample_to_img(mask, image,interpolation='nearest')
    if coords == None:
        coords = get_coordinate(image,mask)
    # mni_coords = apply_affine(image.affine, coords)
    
    datashape = image.shape
    mask,loc = sphere_roi(coords,radius,label,datashape=datashape,data=image.get_fdata())
    
    roi_data = np.zeros_like(image.get_fdata())
    for coord in loc:
        roi_data[coord[0],coord[1],coord[2]] = 1
    roi_img = new_img_like(image, roi_data)
    roi_img.to_filename(savepath)


#%%
# define individual ROI
stats_map_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/training_set/trainsetall/6fold'

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('game1_fmri==1')
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_','-') for p in pid]

ec_roi = r'/mnt/workdir/DCM/docs/Reference/EC_ROI/volume/EC-thr25-2mm.nii.gz'
vmpfc_roi = r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/osfstorage-archive/data/Analysis_ROI_nii/mPFC_roi.nii'

ec_peak_activity = []
vmpfc_peak_activity = []
savedir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/defROI'
for sub in subjects:
    stats_map = opj(stats_map_dir,sub,'ZF_0004.nii')
    ec_savedir = opj(savedir,'EC')
    if not os.path.exists(ec_savedir):
        os.mkdir(ec_savedir)
    vmpfc_savedir = opj(savedir,'vmpfc')
    if not os.path.exists(vmpfc_savedir):
        os.mkdir(vmpfc_savedir)
    ec_savepath = opj(ec_savedir,f'{sub}_EC_func_roi.nii')
    vmpfc_savepath = opj(vmpfc_savedir,f'{sub}_vmpfc_func_roi.nii')
    makeSphereMask(stats_map, ec_roi, ec_savepath, radius=(5/3,5/3,5/3))
    makeSphereMask(stats_map, vmpfc_roi, vmpfc_savepath, radius=(5/3,5/3,5/3))  # coords=(45,89,35)

#%%
# define ROI from F-test group result
task = 'game1'
glm_type = 'M2_Decision'
# EC
stats_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/Setall/' \
            f'group/hp/2ndLevel/_contrast_id_ZF_0011/spmT_0001.nii'
roi = r'/mnt/workdir/DCM/docs/Reference/EC_ROI/volume/EC-thr50-2mm.nii.gz'
savepath = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/defROI/EC/group_EC_func_roi.nii'
makeSphereMask(stats_map, roi, savepath, radius=(5/3,5/3,5/3))

# vmpfc
stats_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/{glm_type}/Setall/' \
            f'group/hp/2ndLevel/_contrast_id_ZF_0011/spmT_0001.nii'
roi = r'/mnt/data/Template/VMPFC_roi.nii'
savepath =f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/defROI/vmpfc/group_vmPFC_func_roi.nii'
makeSphereMask(stats_map, roi, savepath, radius=(5/3,5/3,5/3))

#%%
task = 'game1'
glm_type = 'M2_Decision'
# EC
stats_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/Setall/' \
            f'group/hp/2ndLevel/_contrast_id_ZF_0011/spmT_0001.nii'
roi = r'/mnt/workdir/DCM/docs/Reference/EC_ROI/volume/EC-thr50-2mm.nii.gz'
savepath = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/defROI/EC/check_EC_func_roi.nii'
makeSphereMask(stats_map, roi, savepath, radius=(5/3,5/3,5/3),coords=(33,62,21))
#%%
from nilearn.plotting import view_img

stat_map = load_img( f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/defROI/EC/check_EC_func_roi.nii')

bg_img = load_img(r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz')
vimg = view_img(stat_map_img=stat_map,bg_img=bg_img,threshold=0,vmax=5,symmetric_cbar=1)