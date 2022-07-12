# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:32:48 2021

@author: qyk
"""
import random
import numpy as np
from preparation.simulation import utils,plot
import matplotlib.pyplot as plt
import pandas as pd
from preparation.simulation.plot import plotAngleRadar


def testSimulation():
    angles = np.linspace(0,361,360)
    activation = utils.genSimulateData(angles,0)
    
    plt.plot(np.deg2rad(angles),activation)
    plt.xtick()
    plt.show()
    



#%%
pairRelation = pd.read_excel(r'C:\Myfile\File\工作\PhD\Development cognitive map\experiment\task\dcm_day1\map_set\5x5\map1/pairs_relationship.xlsx')
pairRelation['angle'] = np.rad2deg(np.arctan2(pairRelation['ap_diff'],pairRelation['dp_diff']))
#plotAngleRadar(pairRelation['angle'])
inferPair = pairRelation.query('dp_inference == True')
plotAngleRadar(inferPair['angle'])

#%%
trainAngles = np.linspace(0,359,180)
plot.plotAngleRadar(trainAngles)
plt.show()
omega =0
trainActivation = utils.genSimulateData(trainAngles,0,0.05)
testAngles = np.linspace(1,360,180)
plot.plotAngleRadar(testAngles)
testActivation = utils.genSimulateData(testAngles,0,0.05)
trainData = {'angles':trainAngles,'activation':trainActivation}
testData = {'angles':testAngles,'activation':testActivation}
omega_estimate, beta_estimate = utils.gridCodeParamEsti(trainData,testData,6)

# estimate the gird orientation and beta
columns = ['i_fold','omega', 'omega_estimate', 'beta_estimate']
paramEsti = {'i_fold':[],'omega':[], 'omega_estimate':[], 'beta_estimate':[]}
for i_fold in range(1,13):
    omega_estimate, beta_estimate = utils.gridCodeParamEsti(trainData,testData,i_fold)
    for column in columns:
        paramEsti[column].append(eval(column))
paramEsti_df = pd.DataFrame(paramEsti)
paramEsti_df.set_index('i_fold',inplace=True)

# # output and plot the result
plot.plot6foldSpecificity(paramEsti_df)
testAngles = testData['angles']
testActivation = testData['activation']
omega_estimate = paramEsti_df.at[6,'omega_estimate']
plot.alignVSmisalign(testAngles, testActivation, omega_estimate)
print('Omega:',0,'Omega_estimate:',omega_estimate)
