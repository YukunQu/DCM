import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def read_stats(filepath):
    # read subjects stats file
    # Initialize an empty dictionary
    data = {}
    # Open the file
    with open(filepath, 'r') as file:
        # Read the file line by line
        for line in file:
            # Remove leading and trailing whitespaces
            line = line.strip()
            # Find the lines that start with a digit (these are the data lines)
            if re.match(r'\d', line):
                # Split the line into columns
                columns = line.split()
                # Save StructName and Mean into the dictionary
                # Here, columns[4] is StructName and columns[5] is Mean
                roi_name = columns[4]
                roi_name = roi_name.replace('ctx-','')
                roi_name = roi_name.replace('Left-','lh_')
                roi_name = roi_name.replace('Right-','rh_')
                roi_name = roi_name.replace('lh-','lh_')
                roi_name = roi_name.replace('rh-','rh_')
                data[roi_name] = float(columns[5])
    return data


# get sublist
# load subjects list
qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'

# check if directory exists
if not os.path.isdir(qsiprep_dir):
    raise Exception(f"Directory does not exist: {qsiprep_dir}")

# get sublist
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]

#filter the bad subjects
i = 0
for sub_id in sub_list:
    fd = pd.read_csv(os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_confounds.tsv'), sep='\t')['framewise_displacement']
    mean_fd = np.nanmean(fd)
    if mean_fd > 0.3:
        i += 1
        print(i,sub_id, mean_fd)
        sub_list.remove(sub_id)

sub_list.sort()

dti_metrics = ['fa', 'md']
template = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/{}/dwi/{}_close1.stats'

# Create separate DataFrames for each metric
dfs = {metric: pd.DataFrame() for metric in dti_metrics}

for sub in sub_list:
    for dm in dti_metrics:
        filepath = template.format(sub, dm)
        data = read_stats(filepath)

        # Convert dictionary to pandas Series
        s = pd.Series(data, name=sub)
        # Concatenate the Series to the DataFrame
        dfs[dm] = pd.concat([dfs[dm], s], axis=1)

# Transpose DataFrames and save to separate CSV files
for dm, df in dfs.items():
    df = df.transpose()
    df.index.name = 'sub-id'
    df.to_csv(f'/mnt/workdir/DCM/Result/analysis/structure/MRtrix/aparc_{dm.upper()}_close1_fd0.3.csv')

#%%
# calculate the correlation with the behavior data
import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import pearsonr

# Load the data
df_FA = pd.read_csv('/mnt/workdir/DCM/Result/analysis/structure/MRtrix/metrics/aparc_FA_close1_fd0.3.csv')
df_MD = pd.read_csv('/mnt/workdir/DCM/Result/analysis/structure/MRtrix/metrics/aparc_MD_close1_fd0.3.csv')
df2 = pd.read_csv('/mnt/workdir/DCM/Result/analysis/brain_metrics_game1_20230911.csv')

# Drop columns with NaN values
df_FA = df_FA.dropna(axis=1)
df_MD = df_MD.dropna(axis=1)

# Merge the two dataframes on the subject id
merged_df_FA = pd.merge(df_FA, df2, left_on='sub-id', right_on='Participant_ID')
merged_df_MD = pd.merge(df_MD, df2, left_on='sub-id', right_on='Participant_ID')

# Function to calculate correlation
def calculate_correlation(df, metric,bh_metric):
    lh_aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                                     'fsaverage', "label",
                                     'lh' + ".aparc.annot")
    rh_aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                                 'fsaverage', "label",
                                 'rh' + ".aparc.annot")
    _, _, lh_names = nib.freesurfer.read_annot(lh_aparc_file)
    _, _, rh_names = nib.freesurfer.read_annot(rh_aparc_file)
    names = []
    for name in lh_names:
        name = name.decode('UTF-8')
        if (name == 'unknown') or (name == 'corpuscallosum'):
            continue
        else:
            names.append('lh_'+name)
    for name in rh_names:
        name = name.decode('UTF-8')
        if (name == 'unknown') or (name == 'corpuscallosum'):
            continue
        else:
            names.append('rh_'+name)
    correlations = {'ROI': [], 'lh.'+metric: [], 'rh.'+metric: []}
    for column in names:
        correlations['ROI'].append(column)
        if 'lh' in column:
            correlation_lh, _ = pearsonr(df[column], df[bh_metric])
            correlations['lh.'+metric].append(correlation_lh)
            correlations['rh.'+metric].append(np.nan)
        elif 'rh' in column:
            correlation_rh, _ = pearsonr(df[column], df[bh_metric])
            correlations['lh.'+metric].append(np.nan)
            correlations['rh.'+metric].append(correlation_rh)
        else:
            continue

    # Convert the correlations dictionary to a DataFrame for better visualization
    correlation_df = pd.DataFrame(correlations)

    return correlation_df

# Calculate correlation for both FA and MD
correlation_df_FA = calculate_correlation(merged_df_FA, 'FA','game1_acc')
correlation_df_MD = calculate_correlation(merged_df_MD, 'MD','game1_acc')

# simple test
lh_corr_df = correlation_df_FA[['ROI','lh.FA']].dropna(subset='lh.FA')
rh_corr_df = correlation_df_FA[['ROI','rh.FA']].dropna(subset='rh.FA')
print(pearsonr(lh_corr_df['lh.FA'], rh_corr_df['rh.FA']))

# simple test
lh_corr_df = correlation_df_MD[['ROI','lh.MD']].dropna(subset='lh.MD')
rh_corr_df = correlation_df_MD[['ROI','rh.MD']].dropna(subset='rh.MD')
print(pearsonr(lh_corr_df['lh.MD'], rh_corr_df['rh.MD']))

# Concatenate the correlation DataFrames of FA and MD
final_df = pd.merge(correlation_df_FA, correlation_df_MD,on='ROI')

# print(final_df)
final_df.to_csv('/mnt/workdir/DCM/Result/analysis/structure/MRtrix/aparc_close1_dwi_measures_corr-game1_acc_fd0.3.csv', index=False)
print('finish.')