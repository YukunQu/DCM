# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 19:20:40 2021

@author: QYK
"""
import os
import pandas as pd
import statsmodels.api as sm
from os.path import join as join
from statsmodels.formula.api import ols

#%%
# Sort data
data_dir = r'D:\File\PhD\Development cognitive map\experiment\material\图片库\儿童\标定数据'
like_data = pd.read_excel(join(data_dir,r'Image_rating_data_familiar.xlsx'))
like_score = like_data.iloc[:,2:-2:2].astype(float)
like_score.columns= [str(i) for i in range(1,90)]
#%%
# filter data
low_std_pic = os.listdir(r'D:\File\PhD\Development cognitive map\experiment\material\图片库\儿童\正式图片')
low_std_pic= [pic.split('.')[0] for pic in low_std_pic]
like_score = like_score[low_std_pic]

#%%
# Organize into the form of statistical needs
score = []
image_id = []
sub_id = []

for image in like_score:
    for idx,sub_score in enumerate(like_score[image]):
        score.append(sub_score)
        image_id.append(str(image))
        sub_id.append(str(idx))
        
data_stats = pd.DataFrame({'Score':score,'Image':image_id,'Subject':sub_id})

#%%
model = ols('Score ~ C(Image) + C(Subject) ',data=data_stats).fit()
anova_result = sm.stats.anova_lm(model)
print(anova_result)