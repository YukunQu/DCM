# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:19:35 2021

@author: QYK
"""

import os
import pandas as pd 
import nibabel as nib
from analysis.hexagon.prepare_data import prepare_data
from analysis.hexagon.glm import glm1
from analysis.hexagon.utils import estimateMeanOrientation
from nilearn.image import load_img
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ =="__main__":
    
    run_sessions = r'D:\Data\Development_Cognitive_Map\bids\derivatives\analysis\glm1/run_sessions.xlsx'
    session_condition = pd.read_excel(run_sessions)
    
    save_dir = r'D:\Data\Development_Cognitive_Map\bids\derivatives\analysis\glm1\result'
    
    for row in session_condition.itertuples():
        subj= row.subject
        run = row.run_list
        tr = row.tr    

        func_name = row.func_name
        events_name = row.events_name
        motion_name = row.motion_name
        
        run = run.split(',')
        func_all, design_matrices = prepare_data(subj, run, func_name, 
                                                 events_name, motion_name, tr)  
        design_matrices = design_matrices
        fig,ax = plt.subplots(figsize=(12,8))  
        sns.heatmap(design_matrices.corr(),annot=True)
        plt.title("{}  ; Run :{}-{} ; TR={} ".format(subj,run[0],run[-1],tr),size=36)
        plt.show()