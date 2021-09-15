# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:35:13 2021

@author: qyk
"""

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
import numpy as np
from sklearn.cluster import MeanShift

sns.set(style="darkgrid")

filePath = r'C:\Myfile\File\工作\PhD\Development cognitive map\exp\task\Image_Rating\见数/2021_08_23_09_06_37.xlsx'
rating_data = pd.read_excel(filePath)
rating_data = rating_data.iloc[:,19:-8]

score = rating_data.iloc[1:,0::2].astype(float)
familiar_score = score.iloc[:,:89]
familiar_score.columns= ['Image' + str(i) for i in range(1,90)]
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

pic_like_mean = like_score.mean()
pic_like_std = like_score.std()

# 挑出低被试间方差的图片
pic_sub_std = pic_like_std**2 + pic_familiar_std**2
low_std_pic = pic_sub_std.sort_values()[:35].index.to_list() 
print(len(low_std_pic))

#%%
pic_familiar_like_mean = pd.concat((pic_familiar_mean[low_std_pic],pic_like_mean[low_std_pic]),keys=['familiar_mean','like_mean'],axis=1)
pic_familiar_like_mean = pic_familiar_like_mean.to_numpy()

clustering = MeanShift(bandwidth=0.615).fit(pic_familiar_like_mean)#0.583075
print(clustering.labels_)
print(sum(clustering.labels_>0))
low_substd_imgstd_pic = np.array(low_std_pic)[clustering.labels_==0]
#%%
data_plot = pd.concat([pic_familiar_mean,pic_familiar_std,pic_like_mean,pic_like_std],
                          axis=1,sort=False,keys=['familiar_mean', 'familiar_std', 
                                                  'like_mean','like_std'])
data_plot['selected'] = 0
data_plot.loc[low_substd_imgstd_pic,'selected'] = 1

sns.scatterplot(data=data_plot,x='familiar_mean',y='like_mean',hue='selected')
plt.show()

sns.scatterplot(data=data_plot,x='familiar_std',y='like_std',hue='selected')
plt.show()