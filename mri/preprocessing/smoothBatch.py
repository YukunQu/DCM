# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:40:18 2021

@author: QYK
"""


import os
import pandas as pd 
import nibabel as nib
from analysis.hexagon.prepare_data import smoothData
from nilearn.image import load_img


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
        
        # --------可替换的脚本----------#
        smoothData(subj,run,func_name)