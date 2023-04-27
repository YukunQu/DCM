# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:44:53 2021

@author: qyk
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp.exp_condition.utils import genPairRelation


def plot6foldSpecificity(paramEsti_df):
    sns.set_context(context="talk")
    x = paramEsti_df.index
    y = paramEsti_df['beta_estimate']
    sns.barplot(x=x, y=y, palette="rocket")
    plt.xlabel('i-fold')
    plt.ylabel('Beta estimate')
    plt.show()
    

def alignVSmisalign(testAngles,ytrue_test,omega_estimate):
    alignedD_360 = [(a-omega_estimate)% 360 for a in testAngles]
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if a==13 else a for a in anglebinNum]
    
    label = []
    for binNum in anglebinNum:
        if binNum in range(1,13,2):
            label.append('align')
        elif binNum in range(2,13,2):
            label.append('misalign')
    data = pd.DataFrame({'BinNumber':anglebinNum,'Activation':ytrue_test,
                         'label':label})
    data = data.sort_values(by='BinNumber')
    sns.barplot(data=data,x='BinNumber',
                y='Activation',hue='label',
                # use two color label the 12 bars
                palette=['salmon','grey','salmon','grey','salmon','grey',
                         'salmon','grey','salmon','grey','salmon','grey'],
                )
    plt.tight_layout()
    plt.show()
    

def plotAngleHist(angles):
    angles_set = set(angles)
    bins = np.linspace(-180, 180, len(angles_set))
    plt.hist(angles,bins)
    
    
def plotAngleRadar(angles):
    alignedD_360 = [a % 360 for a in angles]
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if binN == 13 else binN for binN in anglebinNum]
    
   # Compute pie slices
    N = int(360/30)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    binsAngleNum = []
    for binNum in range(1,13):
        binAngleNum = 0
        for a in anglebinNum:
            if a == binNum:
                binAngleNum +=1
        binsAngleNum.append(binAngleNum)
    width = 2*np.pi / (N+1)
    ax = plt.subplot(projection='polar')
    ax.bar(theta, binsAngleNum,width=width,bottom=0.0,  alpha=0.5)
    plt.show() 
    
    
def plotMapAngle(levels=4,plot='radar'):
    img_id = range(1,(levels**2)+1)
    x = range(1,levels+1)
    y = range(1,levels+1)
    p1, p2 = np.meshgrid(x,y)
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    mapData = pd.DataFrame({'Image_ID':img_id,'Attack_Power':p1,'Defence_Power':p2})
    
    # generate the pair data
    pairsData = genPairRelation(mapData)
    pairsData['angle'] = np.rad2deg(np.arctan2(pairsData['ap_diff'],
                                               pairsData['dp_diff']))
    angles = pairsData['angle']
    if plot == 'hist':
        plotAngleHist(angles)
    elif plot == 'radar':
        plotAngleRadar(angles)
        
        
def plotMapsAngleDistribution():
    mapsAngleDistribution = []
    mapsUniqueAngleNum = []
    levels = range(3,12)
    for level in levels:
        angles = plotMapAngle(level)
        uniqueAngleNum = len(set(angles))
        mapsAngleDistribution.append(angles)
        mapsUniqueAngleNum.append(uniqueAngleNum)
        
    fig, axs = plt.subplots(3,3)
    map_n = 0 
    for i in range(3):
        for j in range(3):
            angles = mapsAngleDistribution[map_n]
            uniqueAngleNum = mapsUniqueAngleNum[map_n]
            bins = np.linspace(-180, 180, 358)
            axs[i,j].hist(angles,bins)
            axs[i,j].set_title('{}^2 map, {}'.format(map_n+3,uniqueAngleNum))
            map_n += 1
    
    for ax in axs.flat:
        ax.set(xlabel='angle', ylabel='Number')
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    fig.tight_layout()
    plt.show()