import os

import nilearn.image
import numpy as np
import pandas as pd
from os.path import join as pjoin
from nilearn.image import math_img,load_img

task = 'game2'
# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'{task}_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]

#contrast_1st = ['m2xeudc','decisionxeudc','m2xmanhd','decisionxmanhd','correct_error']
contrast_1st = ['hexagon']

game1_cmap_temp = r'/mnt/data/DCM/result_backup/2023.3.24/Nilearn_smodel/game1/hexagon_spct/Setall/6fold/sub-{}/zmap/{}_zmap.nii.gz'
game2_cmap_temp = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/hexagon_spct/Setall/6fold/sub-{}/zmap/{}_zmap.nii.gz'


for contrast in contrast_1st:
    for sub in sub_list:
        img1 = load_img(game1_cmap_temp.format(sub,contrast))
        img2 = load_img(game2_cmap_temp.format(sub,contrast))
        diff_img = math_img("img2-img1",img1=img1,img2=img2)

        # save the difference image
        save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/hexagon_diff/Setall/6fold/sub-{}/zmap'.format(sub)
        os.makedirs(save_dir, exist_ok=True)
        diff_img.to_filename(os.path.join(save_dir,'{}_diff_zmap.nii.gz'.format(contrast)))

