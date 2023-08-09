import os
import pandas as pd
from nilearn.image import math_img, load_img
from joblib import Parallel, delayed


def cal_diff(sub,game1_cmap_temp,game2_cmap_temp):
    # calculate the neural difference between two games
    img1 = load_img(game1_cmap_temp.format(sub))
    img2 = load_img(game2_cmap_temp.format(sub))
    diff_img = math_img("img2-img1", img1=img1, img2=img2)

    # save the difference image
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/value_diff/Setall/6fold/sub-{}/zmap'.format(sub) # look out
    os.makedirs(save_dir, exist_ok=True)
    diff_img.to_filename(os.path.join(save_dir, '{}_zmap.nii.gz'.format('value')))


# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'game2_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]


game1_cmap_temp = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/value_spct/Setall/6fold/sub-{}/zmap/value_zmap.nii.gz'
game2_cmap_temp = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/value_spct/Setall/6fold/sub-{}/zmap/value_zmap.nii.gz'

# calculate the difference of image between game1 and game2 for each subject parallelly
Parallel(n_jobs=100)(delayed(cal_diff)(sub, game1_cmap_temp, game2_cmap_temp) for sub in sub_list)