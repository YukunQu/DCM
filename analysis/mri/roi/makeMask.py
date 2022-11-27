#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:54:13 2022

@author: QYK
"""
import numpy as np
from nilearn.masking import apply_mask
from nilearn.image import load_img,resample_to_img,new_img_like


def get_coordinate(image, mask):
    # return voxel_wise coordinates of peak point
    mask = resample_to_img(mask,image,interpolation='nearest')
    img_masked = apply_mask(image, mask)
    peak_value = np.max(img_masked)
    print('Peak value:',peak_value)    
    data = image.get_fdata()
    coords = np.where(data==peak_value)
    peak_coords = [coords[0][0], coords[1][0], coords[2][0]]
    return peak_coords


def sphere_roi(voxloc, radius, value, datashape=(91,109,91), data = None):
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
        loc:  sphere roi coordinates
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
        print(coords)

    datashape = image.shape
    mask,loc = sphere_roi(coords,radius,label,datashape=datashape,data=image.get_fdata())
    
    roi_data = np.zeros_like(image.get_fdata())
    for coord in loc:
        roi_data[coord[0],coord[1],coord[2]] = 1
    roi_img = new_img_like(image, roi_data)
    roi_img.to_filename(savepath)


def makeSphereMask_coords(ref_img,savepath,coords,radius=(2,2,2),label=1):
    image = load_img(ref_img)
    datashape = image.shape
    mask,loc = sphere_roi(coords,radius,label,datashape=datashape,data=image.get_fdata())

    roi_data = np.zeros_like(image.get_fdata())
    for coord in loc:
        roi_data[coord[0],coord[1],coord[2]] = 1
    roi_img = new_img_like(image, roi_data)
    roi_img.to_filename(savepath)