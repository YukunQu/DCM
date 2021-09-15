# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:07:54 2021

@author: qyk
"""

"""select the images with least variance of likelity and familiarity 
   between subjects and images"""
   

import os
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

filePath = r'C:\Myfile\File\工作\PhD\Development cognitive map\exp\task\Image_Rating\见数/2021_08_23_09_06_37.xlsx'
rating_data = pd.read_excel(filePath)
rating_data = rating_data.iloc[:,19:-8]

score = rating_data.iloc[1:,0::2].astype(float)
familiar_score = score.iloc[:,:89]
familiar_score.columns= ['Image'+str(i) for i in range(1,90)]
like_score = score.iloc[:,90:]
like_score.columns= ['Image'+str(i) for i in range(1,90)]

rt = rating_data.iloc[1:,1::2].astype(float)
familiar_rt = rt.iloc[:,:89]
like_rt = rt.iloc[:,89:]

# 剔除不认真的被试
familiar_rt  = familiar_rt.astype(float)
familiar_rt_count = familiar_rt[familiar_rt<500].count(axis=1)
familiar_bad_subject = familiar_rt_count[familiar_rt_count>3].index
familiar_score = familiar_score.drop(familiar_bad_subject)

like_rt  = like_rt.astype(float)
like_rt_count= like_rt[like_rt<500].count(axis=1)
like_bad_subject = like_rt_count[like_rt_count>3].index
like_score = like_score.drop(like_bad_subject)

# 计算图片的被试间评分方差
pic_familiar_mean = familiar_score.mean()
pic_familiar_std = familiar_score.std()
sns.scatterplot(x=pic_familiar_mean,y=pic_familiar_std)
plt.show()

pic_like_mean = like_score.mean()
pic_like_std = like_score.std()
sns.scatterplot(x=pic_like_mean,y=pic_like_std)
plt.show()

# 挑出低被试间方差的图片
low_familiar_std_pic = pic_familiar_std.sort_values()[:45].index.to_list()
low_like_std_pic = pic_like_std.sort_values()[:45].index.to_list()
low_std_pic = list(set(low_like_std_pic).intersection(set(low_familiar_std_pic)))
print(len(low_std_pic))

#%%
# 从低被试间方差的图片的组合，计算图片间方差,算出图片间方差最小的图片组合
low_std_pic_groups = list(itertools.combinations(low_std_pic,25))
least_std = 99999
least_std_group = 0
for i, pic_group in enumerate(low_std_pic_groups):
    image_group_failiar_std = pic_familiar_mean[list(pic_group)].std() # 计算图片组合内的方差
    image_group_like_std = pic_like_mean[list(pic_group)].std()
    image_group_std = image_group_failiar_std + image_group_like_std
    print(image_group_std)
    if image_group_std < least_std:
        least_std = image_group_std
        least_std_group = i
least_std_pic_group = low_std_pic_groups[least_std_group]
print(least_std_group)


data_plot = pd.concat([pic_familiar_mean,pic_familiar_std,pic_like_mean,pic_like_std],
                          axis=1,sort=False,keys=['familiar_mean', 'familiar_std', 
                                                  'like_mean','like_std'])
data_plot['selected'] = 0
data_plot.loc[least_std_pic_group ,'selected'] = 1

sns.scatterplot(data=data_plot,x='familiar_mean',y='familiar_std',hue='selected')
plt.show() 

sns.scatterplot(data=data_plot,x='like_mean',y='like_std',hue='selected')
plt.show()