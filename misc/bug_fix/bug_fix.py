# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 00:00:52 2021

@author: -
"""

# bug 修复
import os
import pandas as pd


def reconstruct_map(blockdir,dim):
    # reconstruct map from the training files
    if dim == 'ap':
        rank_diff = 'ap_diff'
        power = 'Attack_Power'
    elif dim == 'dp':
        rank_diff = 'dp_diff'
        power = 'Defence_Power'
    # concat block file
    blockfiles = []
    for i in range(1,21):
        blockpath = os.path.join(blockdir, '{}_Block{}.xlsx'.format(dim,i))
        blockfiles.append(pd.read_excel(blockpath))
    trainCondition = pd.concat(blockfiles,axis=0)
    
    # find the image set
    image_set = set(trainCondition['pic1'])
    
    # find the images with the lowest dim rank
    lowestImage = []
    for image in image_set:
        imageCondition = trainCondition[trainCondition['pic1']==image]
        if len(imageCondition) == 5:
            if imageCondition[rank_diff].sum() == 5:
                lowestImage.append(image)
    rank1 = pd.DataFrame({'Image_ID':lowestImage,power:1})
    
    # find the rank of the other images
    oneLowestImage = lowestImage[0]
    rank2image = trainCondition[trainCondition['pic1']==oneLowestImage]['pic2'].to_list()
    rank2 = pd.DataFrame({'Image_ID':rank2image,power:2})
    
    oneRank2image = rank2image[0]
    imageCondition = trainCondition[trainCondition['pic1']==oneRank2image]
    rank3image = imageCondition[imageCondition[rank_diff]==1]['pic2'].to_list()
    rank3 = pd.DataFrame({'Image_ID':rank3image,power:3})
    
    oneRank3image = rank3image[0]
    imageCondition = trainCondition[trainCondition['pic1']==oneRank3image]
    rank4image = imageCondition[imageCondition[rank_diff]==1]['pic2'].to_list()
    rank4 = pd.DataFrame({'Image_ID':rank4image,power:4})
    
    oneRank4image = rank4image[0]
    imageCondition = trainCondition[trainCondition['pic1']==oneRank4image]
    rank5image = imageCondition[imageCondition[rank_diff]==1]['pic2'].to_list()
    rank5 = pd.DataFrame({'Image_ID':rank5image,power:5})
    dimPower_df= pd.concat([rank1,rank2,rank3,rank4,rank5],axis=0)
    return dimPower_df


blockdir = r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2\train'
dim = 'ap'
ap_df = reconstruct_map(blockdir,dim)
dim = 'dp'
dp_df = reconstruct_map(blockdir,dim)
map_relation = pd.merge(ap_df,dp_df, how='outer',on=['Image_ID'])
map_relation['Image_ID'] = [os.path.basename(idx) for idx in map_relation['Image_ID']]
map_relation.to_csv(r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2_recon/mapSet_relation.csv')