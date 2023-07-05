# RSA simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# sampling angle pairs


def angle_dissimilarity(theta):
    deltaTheta = np.arange(0,60,2)
    theta2 = theta+deltaTheta

    # get angle's activity
    a1 = np.cos(6*np.deg2rad(theta))
    a2 = np.cos(6*np.deg2rad(theta2))
    # calculate the activity dissimilarity of angle paris
    dissimilarity = np.abs(a1 - a2)
    return deltaTheta,dissimilarity


theta_list = np.arange(0, 360, 0.5)
deltaTheta_array = np.zeros(30)
dissimilarity_array = np.zeros((len(theta_list),30))

for i,theta in enumerate(theta_list):
    deltaTheta,dissimilarity = angle_dissimilarity(theta)
    dissimilarity_array[i,:] = dissimilarity

mean_dissimilarity = np.mean(dissimilarity_array,axis=0)
std_dissimilarity = np.std(dissimilarity_array,axis=0)
se_dissimilarity = np.std(dissimilarity_array, axis=0) / np.sqrt(dissimilarity_array.shape[0])

# Plotting line plot
plt.errorbar(deltaTheta, mean_dissimilarity, yerr=std_dissimilarity, fmt='-', capsize=5, elinewidth=2, markeredgewidth=2)