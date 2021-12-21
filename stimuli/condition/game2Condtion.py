# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 23:46:18 2021

@author: qyk
"""
import os 
import numpy as np
import pandas as pd
from math import ceil
from condition.utils import rank_diff,uniformSampling


def labelTrain(df,dim,returnItem):
    rank_diff = df['{}_diff'.format(dim)]
    if rank_diff == 0.5:
        train_label = True
        correctAns = 2
    elif rank_diff == -0.5:
        train_label = True
        correctAns = 1 
    else:
        train_label = False
        correctAns = None
    if returnItem ==  'label':
        return train_label
    elif returnItem == 'Ans':
        return correctAns
    

def splitFarNear(mapFile,newMonsterPoints):
    # find the monsters around the new monsters
    nms_train = []
    nms_test = []
    for nm, coordinates in newMonsterPoints.items():
        aroundPoints = []
        aroundPoints.append([int(coordinates[0]),int(coordinates[1])])
        aroundPoints.append([ceil(coordinates[0]),int(coordinates[1])])
        aroundPoints.append([ceil(coordinates[0]),ceil(coordinates[1])])
        aroundPoints.append([int(coordinates[0]),ceil(coordinates[1])])
        
        aroundImgs = []
        for point in aroundPoints:
            x = point[0]
            y = point[1]
            aroundImg = mapFile.query('(Attack_Power=={})&(Defence_Power=={})'.format(y,x))['Image_ID'].iloc[0]
            aroundImgs.append(aroundImg)
        
        imgsSet = [str(idx)+'.png' for idx in range(1,26)]
        remoteImgs = []
        for img in imgsSet:
            if img not in aroundImgs:
                remoteImgs.append(img)
                
        # combination the new monster and old 
        mapFile.loc[25] = [25,nm,coordinates[1],coordinates[0]]
        # train
        train_pic1 = [nm for img in aroundImgs] + [img for img in aroundImgs]
        train_pic2 = [img for img in aroundImgs] + [nm for img in aroundImgs]
        pic1_ap = [mapFile[mapFile['Image_ID']==p]['Attack_Power'].iloc[0] for p in train_pic1]
        pic1_dp = [mapFile[mapFile['Image_ID']==p]['Defence_Power'].iloc[0] for p in train_pic1]
        pic2_ap = [mapFile[mapFile['Image_ID']==p]['Attack_Power'].iloc[0] for p in train_pic2]
        pic2_dp = [mapFile[mapFile['Image_ID']==p]['Defence_Power'].iloc[0] for p in train_pic2]
        train_df = pd.DataFrame({'pairs_id':range(1,len(train_pic1)+1),
                                 'pic1':train_pic1,'pic2':train_pic2,
                                'pic1_ap':pic1_ap,'pic1_dp':pic1_dp,
                                'pic2_ap':pic2_ap,'pic2_dp':pic2_dp})
        
        # test
        test_pic1 = [nm for img in remoteImgs] + [img for img in remoteImgs]
        test_pic2 = [img for img in remoteImgs] + [nm for img in remoteImgs]      
        pic1_ap = [mapFile[mapFile['Image_ID']==p]['Attack_Power'].iloc[0] for p in test_pic1]
        pic1_dp = [mapFile[mapFile['Image_ID']==p]['Defence_Power'].iloc[0] for p in test_pic1]
        pic2_ap = [mapFile[mapFile['Image_ID']==p]['Attack_Power'].iloc[0] for p in test_pic2]
        pic2_dp = [mapFile[mapFile['Image_ID']==p]['Defence_Power'].iloc[0] for p in test_pic2]
        test_df = pd.DataFrame({'pairs_id':range(1,len(test_pic1)+1),
                                 'pic1':test_pic1,'pic2':test_pic2,
                                'pic1_ap':pic1_ap,'pic1_dp':pic1_dp,
                                'pic2_ap':pic2_ap,'pic2_dp':pic2_dp})       
        nms_train.append(train_df)
        nms_test.append(test_df)
    
    nms_train_df = pd.concat(nms_train,axis=0)
    nms_test_df = pd.concat(nms_test,axis=0)
    
    return nms_train_df,nms_test_df


def game2Train(train_df,mapDir):
    # rank difference
    train_df['ap_diff'] = train_df.apply(rank_diff, axis=1,args=('ap',))
    train_df['dp_diff'] = train_df.apply(rank_diff, axis=1,args=('dp',))
    train_df.loc[:,'pic1'] = 'Image_pool/game1/'+ train_df['pic1']
    train_df.loc[:,'pic2'] = 'Image_pool/game1/'+ train_df['pic2']

    dims = ['ap', 'dp']
    for dim in dims:
        train_df['correctAns'] = train_df.apply(labelTrain,axis=1,args=(dim,'Ans'))
        train_df = train_df.sample(frac=1)
        train_df.to_excel(os.path.join(mapDir,'game2_{}_train.xlsx'.format(dim)),
                           index=False)


def game2Test(test_df,mapDir):
    # rank difference
    test_df['ap_diff'] = test_df.apply(rank_diff, axis=1,args=('ap',))
    test_df['dp_diff'] = test_df.apply(rank_diff, axis=1,args=('dp',))
    test_df['angles'] = np.rad2deg(np.arctan2(test_df['ap_diff'],test_df['dp_diff']))
    angles =test_df['angles'].tolist()
    # use binNum label each angle 
    alignedD_360 = [a % 360 for a in angles]
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if binN == 13 else binN for binN in anglebinNum]
    test_df['bin'] = anglebinNum
    print(len(test_df))
    
    # sample meg pairs
    sampleAngleSet,_ = uniformSampling(test_df,7)
    sampleAngleSet = sampleAngleSet.sample(frac=1)
    fightRule = ['1A2D']*42 + ['1D2A']*42
    sampleAngleSet['fightRule'] = fightRule
    sampleAngleSet = sampleAngleSet.sample(frac=1)

    sampleAngleSet.loc[:,'pic1'] = 'Image_pool/game1/'+ sampleAngleSet['pic1']
    sampleAngleSet.loc[:,'pic2'] = 'Image_pool/game1/'+ sampleAngleSet['pic2']  
    game2test1 = sampleAngleSet[:42].copy()
    game2test2 = sampleAngleSet[42:].copy()
  
    game2test1.to_excel(os.path.join(mapDir,'game2test1.xlsx'),index=False)
    game2test2.to_excel(os.path.join(mapDir,'game2test2.xlsx'),index=False)    
    
    
if __name__ == '__main__':
    levels=5
    setNum= 10
    newMonsterPoints = {'nm1.png':(2.5,2.5),'nm2.png':(2.5,3.5),
                    'nm3.png':(3.5,2.5),'nm4.png':(3.5,3.5)}
    taskDir = r'D:\Research\Development cognitive map\experiment\task\dcm_day4\fMRI\game2_train'
    for mapid in range(1,setNum+1):
        mapDir = os.path.join(taskDir,'map_set','{}x{}','map{}').format(levels,levels,mapid)
        print(mapDir)
        # load map set relationship
        saveDir = os.path.join(mapDir,'game2')
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        mapSetfilePath = os.path.join(mapDir,'mapSet_relation.csv')
        mapFile  = pd.read_csv(mapSetfilePath)
        nms_train_df,nms_test_df = splitFarNear(mapFile,newMonsterPoints)
        game2Train(nms_train_df,saveDir)
        game2Test(nms_test_df,saveDir)
