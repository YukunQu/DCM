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


def uniformSampling(pairs_df,sampleNum):
    # unifrom sampling from every bin
    sampledAngle = [] 
    for i in range(1,13):
        iBinAngle = pairs_df[pairs_df.bin == i]
        iBinSampleAngle = iBinAngle.sample(sampleNum).copy()
        sampledAngle.append(iBinSampleAngle)    
    sampledAngle = pd.concat(sampledAngle,axis=0)
    return sampledAngle


def getBestAngle(pairs_df,sampleNum,sampleTrials):
    minInverseNum = 9999
    bestAngleSet = []
    for i in range(sampleTrials):
        trialNangle = pairs_df.sample(sampleNum).copy()
        repeatNum = 0 
        index = trialNangle.index
        for idx in index:
        	inverseIndex = 600 - idx - 1
        	if inverseIndex in index:
        		repeatNum += 1 
        inverseNum = repeatNum/2      
        if inverseNum < minInverseNum:
            minInverseNum = inverseNum
            bestAngleSet = trialNangle
    return bestAngleSet


def megBlcok(mapDir):
    mapPath = mapDir.split('\\')[-3:]
    blockCondition = ['meg_apBlock1.xlsx','meg_dpBlock1.xlsx',
                      'meg_dpBlock2.xlsx','meg_apBlock2.xlsx']
    blockConditionPath = os.path.join(mapPath[0],mapPath[1],mapPath[3])
    blockCondition = [os.path.join(blockConditionPath,block) 
                      for block in blockCondition]
    blockCondition_df = pd.DataFrame({'blockN':blockCondition})
    blockCondition_df.to_excel(os.path.join(mapDir,'megBlock.xlsx',index=False))
                     
    
def infer1DCondition(pairs_df,mapDir):
    # generate the condition files for 1D inference, which contain ap_test,
    # dp_test, interleaved_dim_test(ap_test+dp_test), ap_fmri, dp_fmri
    pairs_df['angles'] = np.rad2deg(np.arctan2(pairs_df['ap_diff'],pairs_df['dp_diff']))
    angles = pairs_df['angles'].tolist()
    # use binNum label each angle 
    alignedD_360 = [a % 360 for a in angles]
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if binN == 13 else binN for binN in anglebinNum]
    pairs_df['bin'] = anglebinNum
    
    dims = ['ap', 'dp']
    select_column = ['pairs_id','pic1','pic2','ap_diff','dp_diff',
                     'fightRule','correctAns']
    for dim in dims:
        dimInferPair = pairs_df.query('{}_inference == True'.format(dim)).copy()
        dimInferPair['fightRule'] = dim.upper()
        dimInferPair['correctAns'] = dimInferPair.apply(cal_correctAns, axis=1)
        dimInferPair.loc[:,'pic1'] = 'Image_pool/game1/'+ dimInferPair['pic1']
        dimInferPair.loc[:,'pic2'] = 'Image_pool/game1/'+ dimInferPair['pic2']
        
        # sample meg pairs
        sampleAngleSet = getBestAngle(dimInferPair,240,3000) # MEG pairs        
        sampleAngleSet = sampleAngleSet.sample(frac=1)
        megDimBlock1 = sampleAngleSet[:60][select_column]
        megDimBlock2 = sampleAngleSet[60:120][select_column]
        
        megDimBlock1.to_excel(os.path.join(mapDir,'meg_{}Block1.xlsx'.format(dim)),index=False)
        megDimBlock2.to_excel(os.path.join(mapDir,'meg_{}Block2.xlsx'.format(dim)),index=False)
        
        # sample paris for 1D inference test         
        oneDtest1 = sampleAngleSet[120:180][select_column]
        oneDtest2 = sampleAngleSet[180:][select_column]
        oneDtest1.to_excel(os.path.join(mapDir,'{}_1Dtest1.xlsx'.format(dim)),index=False)
        oneDtest2.to_excel(os.path.join(mapDir,'{}_1Dtest2.xlsx'.format(dim)),index=False)
        megBlcok(mapDir)
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


def saveCondition(pairs_df,mapDir,dim):
    select_column = ['pairs_id','pic1','pic2','ap_diff','dp_diff']
    pairs_tmp = pairs_df[select_column]
    pairs_tmp['correctAns'] = pairs_df.apply(labelTrain,axis=1,args=(dim,'Ans'))
    pairs_tmp.loc[:,'pic1'] = 'Image_pool/game1/'+ pairs_tmp['pic1']
    pairs_tmp.loc[:,'pic2'] = 'Image_pool/game1/'+ pairs_tmp['pic2']
    pairs_tmp.to_excel(os.path.join(mapDir,dim+'.xlsx'),index=False)