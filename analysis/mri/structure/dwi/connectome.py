import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# get subjects list
qsirecon_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsirecon'
sub_list = os.listdir(qsirecon_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]

# get adult list
participants_tsv = r'/mnt/workdir/DCM/Result/analysis/brain_metrics_game1_20231025.csv'
data = pd.read_csv(participants_tsv)
data['mean distance code'] = (data['Distance code (Game1-mPFC)'] + data['Distance code (Game2-mPFC)'])/2
data['Grid-like code (mPFC)'] = (data['CV2_Grid-like code (Game1-mPFC)'] + data['Map-alignment2 (Game2-mPFC)'])/2
file_tmp = '{}_dir-PA_space-T1w_desc-preproc_DMN_atlas_sift_invnodevol_radius2_countconnectome.csv'

subs_connectome = np.zeros((len(sub_list), 5, 5))
adults_connectome = np.zeros((5, 5))

adult_connectome_num = 0
age_list = []
acc_list = []
distance_code = []
grid_code = []
for i,s in enumerate(sub_list):
    sub_connectome_path = os.path.join(qsirecon_dir,s,'dwi','connectome2tck',file_tmp.format(s))
    sub_connectome = np.genfromtxt(sub_connectome_path, delimiter=',')
    subs_connectome[i,:,:] = sub_connectome

    age_list.append(data.query(f'Participant_ID=="{s}"')['Age'].values[0])
    acc_list.append(data.query(f'Participant_ID=="{s}"')['game1_acc'].values[0])
    distance_code.append(data.query(f'Participant_ID=="{s}"')['Distance code (Game2-mPFC)'].values[0])
    grid_code.append(data.query(f'Participant_ID=="{s}"')['Map-alignment (Game2-EC)'].values[0])

average_connectome = np.mean(subs_connectome, axis=0)
#%%
# Column names and row indices
columns = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
indices = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']

# Convert the list of lists to a pandas DataFrame with specified column names and row indices
average_connectome = pd.DataFrame(average_connectome, columns=columns, index=indices)
adults_connectome = pd.DataFrame(adults_connectome, columns=columns, index=indices)

# Mask the upper triangular part of the DataFrame
mask = np.triu(np.ones_like(average_connectome, dtype=bool))
for i in range(len(mask)):
    mask[i][i] = False

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(average_connectome, annot=True, cmap='rocket_r', linewidths=.5, fmt=".2f",mask = mask,vmax=100000)
plt.title(f'Average connectome({len(sub_list)})')
plt.show()

#%%
def p2sign(p):
    if  p<=0.001:
        sign='***'
    elif p<=0.01:
        sign='**'
    elif p<=0.05:
        sign='*'
    else:
        sign=''
    return sign

# calculate correlation
from scipy.stats import pearsonr

# for age
corr_matrix = np.zeros((5,5))
p_matrix = np.zeros((5,5))
for x in range(5):
    for y in range(5):
        r,p = pearsonr(age_list,subs_connectome[:,x,y])
        corr_matrix[x,y] = r
        p_matrix[x,y] = p

# Column names and row indices
columns = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
indices = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
# Convert the list of lists to a pandas DataFrame with specified column names and row indices
corr_matrix = pd.DataFrame(corr_matrix, columns=columns, index=indices)
# Convert the p-values matrix to a pandas DataFrame
p_matrix_df = pd.DataFrame(p_matrix, columns=columns, index=indices).round(3).applymap(p2sign)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# plot the correlation matrix
plt.figure(figsize=(8, 6))
#sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', linewidths=.5, fmt=".2f",mask = mask,vmax=0.8)
ax = sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', linewidths=.5, mask=mask, vmax=0.8, fmt=".2f",cbar=True)
for i in range(len(indices)):
    for j in range(len(columns)):
        if not mask[i][j]:
            text = ax.text(j+0.5, i+0.7, f'{p_matrix_df.iloc[i,j]}', ha='center', va='center', color='black', fontsize=12, fontweight='bold',alpha=0.5)
plt.title(f'Correlation matrix(Age)',size=16)
plt.show()

#%%
# for accuracy

# Calculate the mean of the non-NaN values
mean_value = np.nanmean(acc_list)

# Fill the NaN values with the mean value
acc_list = [mean_value if np.isnan(x) else x for x in acc_list]

corr_matrix = np.zeros((5,5))
p_matrix = np.zeros((5,5))
for x in range(5):
    for y in range(5):
        r,p = pearsonr(acc_list,subs_connectome[:,x,y])
        corr_matrix[x,y] = r
        p_matrix[x,y] = p

# Column names and row indices
columns = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
indices = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
# Convert the list of lists to a pandas DataFrame with specified column names and row indices
corr_matrix = pd.DataFrame(corr_matrix, columns=columns, index=indices)
# Convert the p-values matrix to a pandas DataFrame
p_matrix_df = pd.DataFrame(p_matrix, columns=columns, index=indices).round(3).applymap(p2sign)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# plot the correlation matrix
plt.figure(figsize=(8, 6))
#sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', linewidths=.5, fmt=".2f",mask = mask,vmax=0.8)
ax = sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', mask= mask, linewidths=.5, vmax=0.8, fmt=".2f",cbar=True)
for i in range(len(indices)):
    for j in range(len(columns)):
        if not mask[i][j]:
            text = ax.text(j+0.5, i+0.7, f'{p_matrix_df.iloc[i,j]}', ha='center', va='center', color='black', fontsize=12, fontweight='bold',alpha=0.5)
plt.title(f'Correlation matrix(Game1 accuracy)',size=16)
plt.show()
#%%

# For distance code

# Calculate the mean of the non-NaN values
mean_value = np.nanmean(distance_code)

# Fill the NaN values with the mean value
distance_code = [mean_value if np.isnan(x) else x for x in distance_code]

corr_matrix = np.zeros((5,5))
p_matrix = np.zeros((5,5))
for x in range(5):
    for y in range(5):
        r,p = pearsonr(distance_code,subs_connectome[:,x,y])
        corr_matrix[x,y] = r
        p_matrix[x,y] = p

# Column names and row indices
columns = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
indices = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
# Convert the list of lists to a pandas DataFrame with specified column names and row indices
corr_matrix = pd.DataFrame(corr_matrix, columns=columns, index=indices)
# Convert the p-values matrix to a pandas DataFrame
p_matrix_df = pd.DataFrame(p_matrix, columns=columns, index=indices).round(3).applymap(p2sign)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# plot the correlation matrix
plt.figure(figsize=(8, 6))
#sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', linewidths=.5, fmt=".2f",mask = mask,vmax=0.8)
ax = sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', mask= mask, linewidths=.5, vmax=0.8, fmt=".2f",cbar=True)
for i in range(len(indices)):
    for j in range(len(columns)):
        if not mask[i][j]:
            text = ax.text(j+0.5, i+0.7, f'{p_matrix_df.iloc[i,j]}', ha='center', va='center', color='black', fontsize=12, fontweight='bold',alpha=0.5)
plt.title(f'Correlation matrix(distance code)',size=16)
plt.show()

#%%

# For grid-like code

# Calculate the mean of the non-NaN values
mean_value = np.nanmean(grid_code)

# Fill the NaN values with the mean value
grid_code = [mean_value if np.isnan(x) else x for x in grid_code]

corr_matrix = np.zeros((5,5))
p_matrix = np.zeros((5,5))
for x in range(5):
    for y in range(5):
        r,p = pearsonr(grid_code,subs_connectome[:,x,y])
        corr_matrix[x,y] = r
        p_matrix[x,y] = p

# Column names and row indices
columns = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
indices = ['HC', 'LOFC', 'PCC', 'EC', 'mPFC']
# Convert the list of lists to a pandas DataFrame with specified column names and row indices
corr_matrix = pd.DataFrame(corr_matrix, columns=columns, index=indices)
# Convert the p-values matrix to a pandas DataFrame
p_matrix_df = pd.DataFrame(p_matrix, columns=columns, index=indices).round(3).applymap(p2sign)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# plot the correlation matrix
plt.figure(figsize=(8, 6))
#sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', linewidths=.5, fmt=".2f",mask = mask,vmax=0.8)
ax = sns.heatmap(corr_matrix, annot=True, cmap='rocket_r', mask= mask, linewidths=.5, vmax=0.8, fmt=".2f",cbar=True)
for i in range(len(indices)):
    for j in range(len(columns)):
        if not mask[i][j]:
            text = ax.text(j+0.5, i+0.7, f'{p_matrix_df.iloc[i,j]}', ha='center', va='center', color='black', fontsize=12, fontweight='bold',alpha=0.5)
plt.title(f'Correlation matrix(grid-like code)',size=16)
plt.show()