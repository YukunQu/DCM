import os
import pandas as pd
from nilearn.image import math_img, load_img
from joblib import Parallel, delayed



def cal_diff(sub,contrast, game1_cmap_temp, game2_cmap_temp):
    # calculate the neural difference between two games
    img1 = load_img(game1_cmap_temp.format(sub, contrast))
    img2 = load_img(game2_cmap_temp.format(sub, contrast))
    diff_img = math_img("img2-img1", img1=img1, img2=img2)

    # save the difference image
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/base_diff/Setall/6fold/sub-{}/zmap'.format(sub) # look out
    os.makedirs(save_dir, exist_ok=True)
    diff_img.to_filename(os.path.join(save_dir, '{}_zmap.nii.gz'.format(contrast)))


# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'game2_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]

task = 'base_spct'
task_contrast = {'base_spct': ['M1', 'M2_corr', 'decision_corr', 'correct_error'],
                 'hexagon_spct': ['M1', 'M2_corr', 'decision_corr','correct_error','hexagon'],
                 'distance_spct': ['M1', 'M2_corr', 'decision_corr', 'correct_error', 'M2_corrxdistance'],
                 '2distance_spct': ['M1', 'M2_corr', 'decision_corr','m2xeucd', 'decisionxeucd', 'm2xmanhd', 'decisionxmanhd', 'correct_error'],
                 'value_spct': ['M1', 'M2_corr', 'decision_corr', 'correct_error', 'value'],
                 'grid_rsa_corr_trials': ['rsa_ztransf_img_coarse_6fold']}
contrast_1st = task_contrast[task]

game1_cmap_temp = rf'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/{task}'+'/Setall/6fold/sub-{}/zmap/{}_zmap.nii.gz'
game2_cmap_temp = rf'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/{task}'+'/Setall/6fold/sub-{}/zmap/{}_zmap.nii.gz'

for contrast in contrast_1st:
    print(task,'-',contrast,'start.')
    # calculate the difference of image between game1 and game2 for each subject parallelly
    Parallel(n_jobs=100)(delayed(cal_diff)(sub, contrast, game1_cmap_temp, game2_cmap_temp) for sub in sub_list)
