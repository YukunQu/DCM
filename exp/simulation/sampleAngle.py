# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:42:57 2021

@author: qyk
"""

import os 
import numpy as np
import pandas as pd
from simulation.plot import plotAngleRadar
#%%
bestSampleAngle = 0
maxInferNum = 0

pairsFile = r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map1/pairs_relationship.xlsx'
pairs_df = pd.read_excel(pairsFile)
for x in range(1000):
    ntrials = 252
    angleNumPerbin = int(ntrials / 12)
    pairs_df['angles'] = np.rad2deg(np.arctan2(pairs_df['ap_diff'],pairs_df['dp_diff']))
    fmriPair = pairs_df.query('(dp_inference == True)|(ap_inference == True)').copy()
    angles = fmriPair['angles'].tolist()
    
    # use binNum label each angle 
    alignedD_360 = [a % 360 for a in angles]
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if binN == 13 else binN for binN in anglebinNum]
    fmriPair['bin'] = anglebinNum
    
    # drop the point in (1,1) or (5,5)
    dropPair = fmriPair.query('((pic1_ap==1)&(pic1_dp==1))|((pic1_ap==5)&(pic1_dp==5))|((pic2_ap==1)&(pic2_dp==1))|((pic2_ap==5)&(pic2_dp==5))')
    drop_index = dropPair.index
    fmriPair_droped  = fmriPair.drop(drop_index,axis=0)
    
    # sampling from each bin
    sampledAngle = [] 
    for i in range(1,13):
        iBinAngle = fmriPair_droped[fmriPair_droped.bin == i]
        iBinSampleAngle = iBinAngle.sample(angleNumPerbin).copy()
        sampledAngle.append(iBinSampleAngle)    
    sampledAngle = pd.concat(sampledAngle,axis=0)
    #plotAngleRadar(sampledAngle['angles'])
    
    # calcaute the number of 2D inference trials and find the best angle set.
    infer2DNum = len(sampledAngle.query('(ap_inference == True)and(dp_inference == True)'))
    if infer2DNum > maxInferNum:
        bestSampleAngle = sampledAngle
        maxInferNum = infer2DNum
        print(maxInferNum)


#%%
# generate the condition for game1 training
def uniformSampling(pairs_df,sampleNum):
    # unifrom sampling from every bin
    sampledAngle = [] 
    for i in range(1,13):
        iBinAngle = pairs_df[pairs_df.bin == i]
        iBinSampleAngle = iBinAngle.sample(sampleNum).copy()
        sampledAngle.append(iBinSampleAngle)    
    sampledAngle = pd.concat(sampledAngle,axis=0)
    return sampledAngle

pairsFile = r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map1/pairs_relationship.xlsx'
pairsData = pd.read_excel(pairsFile)
pairsData['angles'] = np.rad2deg(np.arctan2(pairsData['ap_diff'],pairsData['dp_diff']))
pairsData.set_index('pairs_id',inplace=True)

samplePair = r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\code\DCM\simulation\sampleResult/sampleAngle_droped.xlsx'
samplePairData = pd.read_excel(samplePair)
samplePairData.set_index('pairs_id',inplace=True)

sampleIndex = samplePairData.index.tolist()
inverseIndex = [600 - idx +1 for idx in sampleIndex]
drop_fmriIndex = sampleIndex + inverseIndex

fmriPair = pairsData.query('(dp_inference == True)|(ap_inference == True)').copy()
# use binNum labeling each angle 
angles = fmriPair['angles'].tolist()
alignedD_360 = [a % 360 for a in angles]
anglebinNum = [round(a/30)+1 for a in alignedD_360]
anglebinNum = [1 if binN == 13 else binN for binN in anglebinNum]
fmriPair['bin'] = anglebinNum
    
fmriPair4test = fmriPair.drop(drop_fmriIndex,axis=0)
plotAngleRadar(fmriPair4test['angles'])
fmriPair4test1 = uniformSampling(fmriPair4test,2)
plotAngleRadar(fmriPair4test1['angles'])
fmriPair4test = fmriPair.drop(fmriPair4test1.index,axis=0)
fmriPair4test2 = uniformSampling(fmriPair4test,2)
plotAngleRadar(fmriPair4test2['angles'])

# save 
fmriPair4test1.to_excel('C:\Myfile\File\工作\PhD\Development cognitive map\experiment\code\DCM\simulation\sampleResult/fmriPair4test1.xlsx')
fmriPair4test2.to_excel('C:\Myfile\File\工作\PhD\Development cognitive map\experiment\code\DCM\simulation\sampleResult/fmriPair4test2.xlsx')