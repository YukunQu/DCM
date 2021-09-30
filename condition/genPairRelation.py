# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 22:48:29 2021

@author: qyk
"""

import os 
from condition import utils
from condition.game1Condtion import game1Condtion

imagePoolDir = r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\task\dcm_day1\Image_pool\game1'
mapSetsDir =  r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\task\dcm_day1\map_set'
levels=5
setNum=10

mapSetsDir = os.path.join(mapSetsDir,'{}x{}'.format(levels,levels))
if not os.path.exists(mapSetsDir):
    os.mkdir(mapSetsDir)


for mapid in range(1,setNum+1):
    mapDir = os.path.join(mapSetsDir,'map{}'.format(mapid))
    mapFile = utils.genMapSet(imagePoolDir,mapDir,levels)
    pairs_df = utils.genPairRelation(mapFile,mapDir)
    pairs_df = utils.genTrainCondition(pairs_df,mapDir)
    pairs_df = utils.infer1DCondition(pairs_df,mapDir)
    pairs_df.to_excel(os.path.join(mapDir,'pairs_relationship.xlsx'),index=False)
    # game1Condtion(pairs_df,mapDir)
