# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:09:56 2021

@author: QYK
"""
import shutil
from os.path import join as join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift

sns.set(style="darkgrid")
#%%
# read data
data_dir = r'D:\File\PhD\Development cognitive map\experiment\material\图片库\儿童\标定数据'
like_data = pd.read_excel(join(data_dir,r'Image_rating_data_like.xlsx'))
familiar_data = pd.read_excel(join(data_dir,r'Image_rating_data_familiar.xlsx'))
#%%
# filter data using RT
like_rt = like_data.iloc[:,3:-2:2].astype(float)
familiar_rt = familiar_data.iloc[:,3:-2:2].astype(float)

familiar_rt_count = familiar_rt[familiar_rt<500].count(axis=1)
familiar_bad_subject = familiar_rt_count[familiar_rt_count>5].index

like_rt  = like_rt.astype(float)
like_rt_count= like_rt[like_rt<500].count(axis=1)
like_bad_subject = like_rt_count[like_rt_count>5].index

like_score = like_data.iloc[:,2:-2:2].astype(float)
familiar_score =  familiar_data.iloc[:,2:-2:2].astype(float)

# drop the score from bad subject
familiar_score = familiar_score.drop(familiar_bad_subject)
like_score = like_score.drop(like_bad_subject)

#%% 
# select images by variance between subjects and within subjects(between images)
# 计算被试间评分方差
like_sub_var = like_score.std()
familiar_sub_var = familiar_score.std()

# 挑出低被试间方差的图片
sub_std = familiar_sub_var**2 + like_sub_var**2
low_sub_std_pic = sub_std.sort_values()[:45].index.to_list() 
print(len(low_sub_std_pic))

#%%
# 使用meanshift算法选择熟悉度和喜爱度接近的图片
pic_familiar_mean = familiar_score.mean()
pic_like_mean = like_score.mean()
pic_low_sub_std_mean = pd.concat((pic_familiar_mean[low_sub_std_pic],pic_like_mean[low_sub_std_pic]),keys=['familiar_mean','like_mean'],axis=1)
pic_low_sub_std_mean = pic_low_sub_std_mean.to_numpy()

clustering = MeanShift(bandwidth=0.5,max_iter=1000).fit(pic_low_sub_std_mean)#0.583075
print(clustering.labels_)
print(sum(clustering.labels_>0))
low_substd_imgstd_pic = np.array(low_sub_std_pic)[clustering.labels_==0]
print(len(low_substd_imgstd_pic))
#%% 
data_plot = pd.concat([pic_familiar_mean,familiar_sub_var,pic_like_mean,like_sub_var],
                          axis=1,sort=False,keys=['familiar_mean', 'familiar_std', 
                                                  'like_mean','like_std'])
data_plot['selected'] = 0
data_plot.loc[low_substd_imgstd_pic,'selected'] = 1

sns.scatterplot(data=data_plot,x='familiar_mean',y='like_mean',hue='selected')
plt.show()

sns.scatterplot(data=data_plot,x='familiar_std',y='like_std',hue='selected')
plt.show()

#%%
# find the image 
low_substd_imgstd_pic = list(low_substd_imgstd_pic)
low_substd_imgstd_pic = [pic[2:] for pic in low_substd_imgstd_pic]

ori_dir = r'D:\File\PhD\Development cognitive map\experiment\material\图片库\儿童\见数标定图库'
dest_dir = r'D:\File\PhD\Development cognitive map\experiment\material\图片库\儿童\筛选3_图片'

for pic in low_substd_imgstd_pic:
    ori_path = join(ori_dir,'{}.png'.format(pic))
    dest_path = join(dest_dir,'{}.png'.format(pic))
    shutil.copy(ori_path, dest_path)