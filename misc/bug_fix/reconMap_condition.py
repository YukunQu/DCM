# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:18:58 2021

@author: QYK
"""

import os 
import pandas as pd 
from condition import utils
from condition.game1Condtion import game1Condtion


# mapDir = r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2'
# mapFile = pd.read_csv(r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2_recon/mapSet_relation.csv')
# pairs_df = utils.genPairRelation(mapFile,mapDir)
# pairs_df = utils.genTrainCondition(pairs_df,mapDir)
# pairs_df = utils.infer1DCondition(pairs_df,mapDir)
# pairs_df.to_excel(os.path.join(mapDir,'pairs_relationship.xlsx'),index=False)

pairs_df = pd.read_excel(r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map2_recon/pairs_relationship.xlsx')
taskDir = r'D:\File\PhD\Development cognitive map\experiment\task\dcm_day1'
middle_path = r'map_set\5x5\map2_recon'
game1Condtion(taskDir,middle_path,pairs_df)