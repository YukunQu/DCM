import os
import numpy as np
import pandas as pd

# get subject list
qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'
qsirecon_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsirecon'
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]
sub_list.sort()
sub_list = sub_list

#filter the bad subjects
i = 0
for sub_id in sub_list:
    fd = pd.read_csv(os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_confounds.tsv'), sep='\t')['framewise_displacement']
    mean_fd = np.nanmean(fd)
    if mean_fd > 0.5:
        i += 1
        print(i,sub_id, mean_fd)
        sub_list.remove(sub_id)


# move the FA file into a new directory for good subjects
for sub_id in sub_list:
    fa_file = os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dwi_FA.nii.gz')
    os.system(f'cp {fa_file} /mnt/workdir/DCM/BIDS/derivatives/TBSS/FA')
