# -*- coding: utf-8 -*-
"""
fMRI experiment simulation to find the appropriate sampling angle

Created on Tue Sep 14 18:58:34 2021

@author: qyk

"""
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm


def samplingAngle(levels,ntrials):
    img_id = range(1,(levels**2)+1)
    x = range(1,levels+1)
    y = range(1,levels+1)
    p1, p2 = np.meshgrid(x,y)
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    mapData = pd.DataFrame({'img_id':img_id,'ap':p1,'dp':p2})
    
    # generate the simulation-angles data
    angles = []
    m1s = []
    m2s = []
    for n in range(ntrials):
        m1 = mapData.sample(1) # monster 1 as the start
        m2 = mapData.sample(1) # monster 2 as the end
        x_dis = m2.iloc[0]['dp'] - m1.iloc[0]['dp']
        y_dis = m2.iloc[0]['ap'] - m1.iloc[0]['ap']
        angle = np.rad2deg(np.arctan2(y_dis, x_dis)) # -pi : pi
        angles.append(angle)
        m1s.append((m1['dp'], m1['ap']))
        m2s.append((m2['dp'], m2['ap']))
    return angles,m1s,m2s


def genSimulateData(angles,omega=None,ampNoise=0.1):
    if omega == None:
        omegas = range(-29,30,5) # 根据Erie 的代码重新改范围
        omega = random.choice(omegas) 
    
    y_true = []
    for angle in angles:
        noise = ampNoise * np.random.rand()
        y_true.append(1 + 1*np.cos(np.deg2rad(6*(angle - omega))) + noise)
    return y_true


def gridCodeParamEsti(trainData,testData,ifold):
    # estimate the omega
    trainAngles = trainData['angles']
    x0 = np.ones(len(trainAngles))
    x1 = np.sin(np.deg2rad([ifold*a for a in trainAngles]))
    x2 = np.cos(np.deg2rad([ifold*a for a in trainAngles]))
    X= np.stack((x0,x1,x2),axis=1)
    betaTrain = np.linalg.pinv(X).dot(trainData['activation'])
    omega_estimate = np.rad2deg(np.arctan2(betaTrain[1],betaTrain[2])/ifold)
        
    # test the omega effect
    x = []
    testAngles = testData['angles']
    for angle in testAngles:
        x.append(np.cos(np.deg2rad(ifold*(angle - omega_estimate))))
    
    x = sm.add_constant(x)
    glm_model = sm.GLM(testData['activation'],x)
    fit_summary = glm_model.fit()
    beta_estimate = fit_summary.params[1]
    return omega_estimate, beta_estimate
