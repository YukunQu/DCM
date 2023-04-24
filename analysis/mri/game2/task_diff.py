# The separation of decision (value-parametric modulation)
# and mental simulation (distance/grid-like coding)  in neural operation

import os
import pandas as pd
from nilearn.image import math_img, load_img
from joblib import Parallel, delayed


def cal_task_diff(sub):
    # set template for task1 and task2
    task1_cmap_temp = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/distance_spct/' \
                      r'Setall/6fold/sub-{}/zmap/M2_corrxdistance_zmap.nii.gz'
    task2_cmap_temp = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/value_spct/' \
                      r'Setall/6fold/sub-{}/zmap/value_zmap.nii.gz'

    # calculate the neural difference between two games
    img1 = load_img(task1_cmap_temp.format(sub))
    img2 = load_img(task2_cmap_temp.format(sub))
    diff_img = math_img("img1-img2", img1=img1, img2=img2)

    # save the difference image
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/distance_value_diff/Setall/6fold/sub-{}/zmap'.format(sub)
    os.makedirs(save_dir, exist_ok=True)
    diff_img.to_filename(os.path.join(save_dir, 'distance_value_diff_zmap.nii.gz'))


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'game1_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]

# calculate the difference of image between task1 and task2 for each subject parallelly
Parallel(n_jobs=100)(delayed(cal_task_diff)(sub) for sub in sub_list)