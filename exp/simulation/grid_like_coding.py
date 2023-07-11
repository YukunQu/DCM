# -*- coding: utf-8 -*-
"""
fMRI experiment simulation to find the appropriate sampling angle

Created on Wed Sep  8 16:00:38 2021

@author: qyk
"""
import random
import pandas as pd
from exp.simulation import utils
from exp.simulation import plot

# setting parameters
levels = 5
ntrials = 252
nsub=20
ampNoise = 1

# grid-like coding simulation #
# sampling the angles
angles,m1s,m2s = utils.samplingAngle(levels,ntrials)

# split data
trainData = {}
testData = {}

trainData['angles'] = angles[:int(len(angles)/2)]
testData['angles'] = angles[int(len(angles)/2):]

# generate the simulation-data
omegas = range(-29,30,5)  # 根据Erie 的代码重新改范围
omega = random.choice(omegas) 
trainData['activation'] = utils.genSimulateData(trainData['angles'],omega,ampNoise)
testData['activation'] = utils.genSimulateData(testData['angles'],omega,ampNoise)

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
plot.plotAngleRadar(angles)
plot.plot6foldSpecificity(paramEsti_df)
testAngles = testData['angles']
testActivation = testData['activation']
omega_estimate = paramEsti_df.at[6,'omega_estimate']
plot.alignVSmisalign(testAngles, testActivation, omega_estimate)
print('Omega:',omega,'Omega_estimate:',omega_estimate)
