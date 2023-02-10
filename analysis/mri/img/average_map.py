import os
import pandas as pd
from nilearn.image import mean_img


m2_zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/distance_m2/Setall/6fold/{}/con_0001.nii'
decision_zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/distance_decision/Setall/6fold/{}/con_0001.nii'
average_zmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/distance_average/Setall/6fold/{}'

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
    average_map.to_filename(os.path.join(amap,'con_0001.nii'))
    print("The {} is done.".format(sub_id))
