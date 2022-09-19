# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:14:13 2021

@author: qyk
"""
import os 
import random
from os.path import join
import numpy as np
import pandas as pd
from condition.utils import uniformSampling


def sampleAngleSet(pairs_df):
    bestSampleAngle = []
    maxInferNum = 0
    
    for x in range(500):
        ntrials = 252
        session = 3 
        angleNumPerbin = int(ntrials/session/12)
        fmriPair = pairs_df.query('(dp_inference == True)|(ap_inference == True)').copy()
               
        # drop the point in (1,1) or (5,5)
        dropPair = fmriPair.query('((pic1_ap==1)&(pic1_dp==1))|((pic1_ap==5)&(pic1_dp==5))|((pic2_ap==1)&(pic2_dp==1))|((pic2_ap==5)&(pic2_dp==5))')
        drop_index = dropPair.index
        fmriPair_droped  = fmriPair.drop(drop_index,axis=0)
        
        # sampling from each bin
        sampleAngleSet1,fmriPair_droped = uniformSampling(fmriPair_droped,angleNumPerbin)
        sampleAngleSet2,fmriPair_droped = uniformSampling(fmriPair_droped,angleNumPerbin)
        sampleAngleSet3,fmriPair_droped = uniformSampling(fmriPair_droped,angleNumPerbin)
        
        # calcaute the number of 2D inference trials and find the best angle set.
        sampleAngle = pd.concat([sampleAngleSet1,sampleAngleSet2,sampleAngleSet3])
        infer2DNum = len(sampleAngle.query('(ap_inference == True)and(dp_inference == True)'))
        if infer2DNum > maxInferNum:
            bestSampleAngle = sampleAngle
            bsetSampleSet = [sampleAngleSet1,sampleAngleSet2,sampleAngleSet3]
            maxInferNum = infer2DNum
            print(maxInferNum)
            
    index = bestSampleAngle.index.to_list()
    inverseIndex  = [600 - idx - 1 for idx in index]
    fmriPairIndex = index + inverseIndex
    pilotPool = fmriPair.drop(fmriPairIndex,axis=0)
    pilotAngle = pilotPool.sample(24).copy()
    return bsetSampleSet, pilotAngle
    
    
def game1Condtion(taskDir,mapDir,pairs_df):
    pairs_df.loc[:,'pic1'] = 'Image_pool/game1/'+ pairs_df['pic1']
    pairs_df.loc[:,'pic2'] = 'Image_pool/game1/'+ pairs_df['pic2']
    bestAngleSet,pilotAngle = sampleAngleSet(pairs_df) # find the best angle set.

    saveDir = join(taskDir,mapDir,'fmri')
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    # random the order of angle pairs and split into 6 blocks
    blockID = 1
    blockTrials= 42
    blockFile = []
    for bestSet in bestAngleSet:
        fightRule = ['1A2D']*42 + ['1D2A']*42
        random.shuffle(fightRule)
        bestSet['fightRule'] = fightRule
        bestSet = bestSet.sample(frac=1)
        pairsIndex = bestSet.index
        for block in [1,2]:
            block_first = int((block-1) * blockTrials)
            block_last = int(block * blockTrials)
            blockPairsIndex = pairsIndex[block_first:block_last]           
            blockPairs = bestSet.loc[blockPairsIndex]     
            # save the block condition file
            blockFileName = 'fmri_Block{}.xlsx'.format(blockID)
            blockID += 1
            savePath = join(saveDir,blockFileName)
            blockPairs.to_excel(savePath,index=False)            
            blockFile.append(join(mapDir,'fmri',blockFileName).replace('\\', '/'))

    pilotAngle = pilotAngle.sample(frac=1) # random the order of angle pairs and split into 6 blocks
    fightRule = ['1A2D']*12 + ['1D2A']*12
    random.shuffle(fightRule)
    pilotAngle['fightRule'] = fightRule
    
    fmriBlocks = pd.DataFrame({'blockN':blockFile})
    fmriBlocks.to_excel(join(saveDir,'fmriBlocks.xlsx'),index=False)
    pilotAngle.to_excel(join(saveDir,'pilotTrials.xlsx'),index=False)
    

if __name__ == '__main__':
    levels=5
    setNum=10
    taskDir = r'D:\Research\Development cognitive map\experiment\task\dcm_day4\fMRI\game1'
    for mapid in range(1,setNum+1):
        mapDir = join('map_set','{}x{}','map{}').format(levels,levels,mapid)
        pairs_df_path = join(taskDir,mapDir,'pairs_relationship.xlsx')
        pairs_df = pd.read_excel(pairs_df_path) 
        game1Condtion(taskDir,mapDir,pairs_df)