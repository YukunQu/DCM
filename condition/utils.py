# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:28:09 2021

@author: qyk
"""

# generate the stimulus sets from stimulus pool #

import os
import random
import itertools
import numpy as np
import pandas as pd


def extract_rank(x,rf,dim):
    return rf.loc[rf['Image_ID']==x][dim].values[0]


def rank_diff(df,dim):
    return df['pic2_{}'.format(dim)] - df['pic1_{}'.format(dim)]


def one_dim_inference(df,dim):
    rank_diff = abs(df['{}_diff'.format(dim)])
    if rank_diff >1:
        oneDimInfer = True
    else:
        oneDimInfer = False
    return oneDimInfer


def labelTrain(df,dim,returnItem):
    rank_diff = df['{}_diff'.format(dim)]
    if rank_diff == 1:
        train_label = True
        correctAns = 2
    elif rank_diff == -1:
        train_label = True
        correctAns = 1 
    else:
        train_label = False
        correctAns = None
    if returnItem ==  'label':
        return train_label
    elif returnItem == 'Ans':
        return correctAns


def genMapSet(imagePoolDir, mapDir,levels):
    # generate the stimulus sets from stimulus pool
    if not os.path.exists(mapDir):
        os.mkdir(mapDir) # create image set directory
    imgs_pool = os.listdir(imagePoolDir)
    mapset = random.sample(imgs_pool, levels**2) # random sampling from image pool
    
    # assign the attack and defence power
    x = range(1,levels+1)
    y = range(1,levels+1)
    d1,d2 = np.meshgrid(x,y)
    d1 = d1.reshape(-1)
    d2 = d2.reshape(-1)
    mapFile = pd.DataFrame({'Image_ID':mapset,'Attack_Power':d1,'Defence_Power':d2})
    mapFile.to_csv(os.path.join(mapDir,'mapSet_relation.csv'))
    return mapFile
        

def genPairRelation(mapFile,mapDir='',save=False):
    # genrate the relationship file of stimulis pairs 
    pairs = list(itertools.combinations(mapFile['Image_ID'],2))
    pic1 = [p[0] for p in pairs] + [p[1] for p in pairs][::-1]
    pic2 = [p[1] for p in pairs] + [p[0] for p in pairs][::-1]
    pairs_id = range(1,len(pic1)+1)
    pairs_df = pd.DataFrame({'pairs_id':pairs_id,'pic1':pic1,'pic2':pic2})

    pairs_df['pic1_ap']= pairs_df['pic1'].apply(extract_rank,args=(mapFile,'Attack_Power'))
    pairs_df['pic1_dp']= pairs_df['pic1'].apply(extract_rank,args=(mapFile,'Defence_Power'))
    pairs_df['pic2_ap']= pairs_df['pic2'].apply(extract_rank,args=(mapFile,'Attack_Power'))
    pairs_df['pic2_dp']= pairs_df['pic2'].apply(extract_rank,args=(mapFile,'Defence_Power'))
    
    # rank difference
    pairs_df['ap_diff'] = pairs_df.apply(rank_diff, axis=1,args=('ap',))
    pairs_df['dp_diff'] = pairs_df.apply(rank_diff, axis=1,args=('dp',))
    
    # dimension inference
    pairs_df['ap_inference'] = pairs_df.apply(one_dim_inference, axis=1,args=('ap',))
    pairs_df['dp_inference'] = pairs_df.apply(one_dim_inference, axis=1,args=('dp',))

    if save == True:
        pairs_df.to_excel(os.path.join(mapDir,'pairs_relation.xlsx'),index=False)
    return pairs_df


def genTrainCondition(pairs_df,mapDir):
    # AP DP train
    select_column = ['pairs_id','pic1','pic2','ap_diff','dp_diff']
    dims = ['ap', 'dp']
    for dim in dims:
        trainDim = '{}_train'.format(dim)
        pairs_df[trainDim] = pairs_df.apply(labelTrain,axis=1,args=(dim,'label'))
        pairs_tmp = pairs_df[pairs_df[trainDim]==True][select_column]
        pairs_tmp['correctAns'] = pairs_df.apply(labelTrain,axis=1,args=(dim,'Ans'))
        pairs_tmp.loc[:,'pic1'] = 'Image_pool/game1/'+ pairs_tmp['pic1']
        pairs_tmp.loc[:,'pic2'] = 'Image_pool/game1/'+ pairs_tmp['pic2']
        pairs_tmp.to_excel(os.path.join(mapDir,'{}_train.xlsx'.format(dim)),
                           index=False)
    return pairs_df


def infer1DCondition(pairs_df,mapDir,save=False):
    # generate the condition files for 1D inference, which contain ap_test,
    # dp_test, interleaved_dim_test(ap_test+dp_test), ap_fmri, dp_fmri
    
    dims = ['ap', 'dp']
    pairs_num = int(len(pairs_df) * 0.5)
    pairs_tophalf = pairs_df[:pairs_num]
    
    for dim in dims:
        infer1DTestIndex = pairs_tophalf[pairs_tophalf['{}_inference'.format(dim)]==1].sample(frac=0.2).index 
        inverse_index = 2 * pairs_num - infer1DTestIndex - 1
        infer1DTestIndex = infer1DTestIndex.tolist() + inverse_index.tolist()
        pairs_df['{}_test'.format(dim)] = pairs_df['{}_train'.format(dim)].copy()
        pairs_df.loc[infer1DTestIndex,'{}_test'.format(dim)] = True
        pairs_df['{}_fmri'.format(dim)] = pairs_df['{}_inference'.format(dim)].copy()
        pairs_df.loc[infer1DTestIndex,'{}_fmri'.format(dim)] = False
        if save == True:
            condition = '{}_test'.format(dim)
            saveCondition(pairs_df, mapDir, condition)
    pairs_df['interleaved_dim_test'] = pairs_df['ap_test'].copy()
    pairs_df.loc[pairs_df.dp_test==True,'interleaved_dim_test'] = True
    if save == True:
        condition = 'interleaved_dim_test'
        saveCondition(pairs_df, mapDir, condition)
    return pairs_df


def cal_correctAns(df):
    if df['fightRule'] == '1A2D':
        diff = df['pic1_ap'] - df['pic2_dp']
        if diff >= 0:
            correctAns = 1
        else:
            correctAns = 2
    elif df['fightRule'] == '1D2A':
        diff = df['pic2_ap'] - df['pic1_dp']
        if diff >= 0:
            correctAns = 2
        else:
            correctAns = 1
    elif df['fightRule'] == 'AP':
        diff = df['pic2_ap'] - df['pic1_ap']
        if diff > 0:
            correctAns = 2
        else:
            correctAns = 1
    elif df['fightRule'] == 'DP':
        diff = df['pic2_dp'] - df['pic1_dp']
        if diff > 0 :
            correctAns = 2
        else:
            correctAns = 1
    return correctAns


def saveCondition(pairs_df,mapDir,condition):
    select_column = ['pairs_id','pic1','pic2','ap_diff','dp_diff']
    if condition != 'interleaved_dim_test':
        pairs_tmp = pairs_df[ pairs_df[condition]==True][select_column]
    else:
        pairs_apTest = pairs_df[pairs_df['ap_test']==True][select_column]
        pairs_apTest['fightRule'] = 'AP'
        pairs_dpTest = pairs_df[pairs_df['dp_test']==True][select_column]
        pairs_dpTest['fightRule'] = 'DP'
        pairs_tmp = pd.concat([pairs_apTest, pairs_dpTest])
    pairs_tmp.loc[:,'pic1'] = 'Image_pool/game1/'+ pairs_tmp['pic1']
    pairs_tmp.loc[:,'pic2'] = 'Image_pool/game1/'+ pairs_tmp['pic2']
    pairs_tmp.to_excel(os.path.join(mapDir,condition+'.xlsx'),index=False)