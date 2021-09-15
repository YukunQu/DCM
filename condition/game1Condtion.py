# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:14:13 2021

@author: qyk
"""
import os 
import pandas as pd
from condition.utils import cal_correctAns


def game1Condtion(pairsR,mapDir):
    # generate the condition files for behaviral training and fmri experiment
    # of Game 1
    # 留待加入训练和核磁实验的condition分离,下面的重复代码可以合并
    # queryRule = ['ap_fmri==1','dp_fmri==1',
    #'(ap_inference==1)&(dp_inference==1)','(ap_inference==1)&(dp_inference==1)']
    # fightRule = ['AP','DP',1A2D',1D2A'] 
    # 弄一个字典
    # for fR in fightRule
    # 1D inference in ap 
    ap_fmri = pairsR.query('ap_fmri==1')
    # game1TrainIndex = ap_fmri.sample(frac=0.05).index
    ap_fmri = ap_fmri.sample(frac=0.05)
    ap_fmri = ap_fmri[['pairs_id','pic1','pic2','pic1_ap','pic1_dp','pic2_ap','pic2_dp']]
    ap_fmri.loc[:,'pic1'] = 'Image_pool/game1/'+ ap_fmri['pic1']
    ap_fmri.loc[:,'pic2'] = 'Image_pool/game1/'+ ap_fmri['pic2']
    ap_fmri['fightRule'] = 'AP'
    
    # 1D inference in dp 
    dp_fmri = pairsR.query('dp_fmri==1')
    dp_fmri = dp_fmri.sample(frac=0.05)
    # game1TrainIndex.append()
    dp_fmri = dp_fmri[['pairs_id','pic1','pic2','pic1_ap','pic1_dp','pic2_ap','pic2_dp']]
    dp_fmri.loc[:,'pic1'] = 'Image_pool/game1/'+ dp_fmri['pic1']
    dp_fmri.loc[:,'pic2'] = 'Image_pool/game1/'+ dp_fmri['pic2']
    dp_fmri['fightRule'] = 'DP'
    
    # 2D inference in two dimension
    mix_fmri = pairsR.query('(ap_inference==1)&(dp_inference==1)')
    mix_fmri = mix_fmri.sample(frac=0.05) # 便捷做法，留待采样仿真后修正
    # game1TrainIndex.append()
    mix_fmri = mix_fmri[['pairs_id','pic1','pic2','pic1_ap','pic1_dp','pic2_ap','pic2_dp']]
    mix_fmri.loc[:,'pic1'] = 'Image_pool/game1/'+ mix_fmri['pic1']
    mix_fmri.loc[:,'pic2'] = 'Image_pool/game1/'+ mix_fmri['pic2']
    
    mix_fmri_1a2d = mix_fmri.copy()
    mix_fmri_1d2a = mix_fmri.copy()
    
    mix_fmri_1a2d['fightRule'] = '1A2D'
    mix_fmri_1d2a['fightRule'] = '1D2A'
    
    game1_fmri = pd.concat((mix_fmri_1a2d,mix_fmri_1d2a,ap_fmri,dp_fmri))
    game1_fmri['correctAns'] = game1_fmri.apply(cal_correctAns, axis=1)
    game1_fmri.to_excel(os.path.join(mapDir,'game1_train.xlsx'),index=False)
    
    # pairR.loc[game1TrainIndex,'game1Train'] = True
    # pairR.loc[game1ExpIndex,'game1Train'] =True
    # return pairR