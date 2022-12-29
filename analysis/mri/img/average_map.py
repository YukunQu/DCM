import os
import pandas as pd
from nilearn.image import mean_img


m2_zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/m2plus_hexagon_correct_trials/Setall/6fold/{}/ZF_0005.nii'
decision_zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/decision_hexagon_correct_trials/Setall/6fold/{}/ZF_0005.nii'
average_zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/average_hexagon_plus_correct_trials/Setall/6fold/{}'

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subjects = data['Participant_ID'].to_list()

for sub_id in subjects:
    m2zmap = m2_zmap_template.format(sub_id)
    dzmap = decision_zmap_template.format(sub_id)
    amap = average_zmap_template.format(sub_id)
    if not os.path.exists(amap):
        os.mkdir(amap)
    average_map = mean_img([m2zmap,dzmap])
    average_map.to_filename(os.path.join(amap,'ZF_0005.nii'))
    print("The {} is done.".format(sub_id))
