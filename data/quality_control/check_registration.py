"""The srcipt detect outliers of statistc across subjects to filter the subject with wrong preprocessing."""

import os

import numpy as np
from nilearn.image import load_img,mean_img,get_data,binarize_img
from nilearn.masking import apply_mask

#%%

#  """ find the subject may have wrong statistic map accoring to specific ROI"""

fmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_separate_phases_correct_trials/Setall/6fold'

sub_list = os.listdir(r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_separate_phases_correct_trials/Setall/6fold')
sub_list = [s for s in sub_list if 'sub-' in s]
sub_list.sort()

fmap_list = [os.path.join(fmap_template,s,'zmap/hexagon_zmap.nii.gz') for s in sub_list]
mask = load_img(r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL_prob.nii.gz')
mask = binarize_img(mask,50)
from nilearn.plotting import plot_roi
plot_roi(mask)
#%%
fmap_masked = apply_mask(fmap_list,mask)
meanf = [maskedf.mean() for maskedf in fmap_masked]

# 利用箱线图检测异常值
import matplotlib.pyplot as plt
plt.boxplot(meanf)
plt.show()

# 利用Z-score检测异常值
from scipy import stats
#zs = np.abs(stats.zscore(meanf))
zs = stats.zscore(meanf)
threshold = 2
for z,sub_id in zip(meanf,sub_list):
    if (z>threshold) or (z<-threshold):
        print(sub_id)
        print(z)


#%%
#  """ find the subject may have wrong statistic map accoring to whole brain voxels"""
sub_list = os.listdir(r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_separate_phases_correct_trials/Setall/6fold')
sub_list = [s for s in sub_list if 'sub-' in s]
sub_list.sort()

map_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_separate_phases_correct_trials/Setall/6fold'
map_list = [os.path.join(map_template,s,'stat_map/hexagon_smap.nii.gz') for s in sub_list]

imgs = [get_data(mean_img(mapx)) for mapx in map_list]
imgs = np.array(imgs)
print("Images loading finished.")
#%%

# Calculate z-scores along the first axis for each value of n
mean = np.mean(imgs, axis=0)
std = np.std(imgs, axis=0)
z_scores = (imgs - mean) / std

# Set threshold for outlier detection
threshold = 3  # F-test:10  t-test:5

# Find indices of outlier values along first axis for each value of n
outlier_indices = []
for i in range(imgs.shape[0]):
    n_z_scores = z_scores[i]
    n_outlier_indices = np.where(np.abs(n_z_scores) > threshold)[0]
    outlier_indices.append(n_outlier_indices)

# Print number of outlier values and their indices for each value of n
# The bad subject will have large numbers of outliers.
for i,sub_id in zip(range(imgs.shape[0]),sub_list):
    n_outliers = outlier_indices[i]
    if len(n_outliers)>0:
        print(f"Number of outliers for {sub_id}: {len(n_outliers)}")
        print(f"Outlier indices for {sub_id}:", n_outliers)
