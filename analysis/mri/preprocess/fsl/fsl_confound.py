import os
import numpy as np
import pandas as pd


def get_fsl_confounds(sub_id, run_id):
    confound_file = rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/{sub_id}/func/' \
                    rf'{sub_id}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/FSL/preprocessed_data/confound'
    save_template = fr'{sub_id}_game1_run-0{run_id}_confounds.txt'
    motions_df = pd.read_csv(confound_file,sep='\t')
    motion_columns = [#'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                       'csf', 'white_matter']
    motions = motions_df[motion_columns].to_numpy()
    motion_outpath = os.path.join(save_dir,save_template)
    np.savetxt(motion_outpath,motions,delimiter='  ')


sub_id = r'sub-079'
runs = range(1,7)

for run_id in runs:
    get_fsl_confounds(sub_id,run_id)