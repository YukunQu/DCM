# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 17:15:47 2021

@author: qyk

"""
import os
from os.path import join
import pandas as pd


def divide2TrainBlock(taskDir,mapDir,blockTrials=10):
    for dim in ['ap','dp']:
        trainCondition = pd.read_excel(join(taskDir,mapDir,'{}_train.xlsx'.format(dim)))
        saveDir = join(taskDir,mapDir,'train')
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
    
        blockFile = []
        pairsNum = len(trainCondition)
        part = int(pairsNum/blockTrials)
        trainCondition = trainCondition.sample(frac=1)
        pairsIndex = trainCondition.index
        for block in range(1,part+1):
            block_first = int((block-1) * blockTrials)
            block_last = int(block * blockTrials)
            blockPairsIndex = pairsIndex[block_first:block_last]           
            blockPairs = trainCondition.loc[blockPairsIndex]     
            # save the block condition file
            blockFileName = '{}_Block{}.xlsx'.format(dim,block)
            savePath = join(saveDir,blockFileName)
            blockPairs.to_excel(savePath,index=False)            
            blockFile.append(join(mapDir,'train',blockFileName).replace('\\', '/'))
        trainBlocks = pd.DataFrame({'blockFile':blockFile})
        trainBlocks.to_excel(join(saveDir,'{}_trainBlocks.xlsx'.format(dim)),index=False)
      
        
if __name__ == '__main__':
    levels=5
    setNum=10
    taskDir = r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\task\dcm_day1'
    for mapid in range(1,setNum+1):
        mapDir = os.path.join('map_set','{}x{}','map{}').format(levels,levels,mapid)
        divide2TrainBlock(taskDir,mapDir,10)