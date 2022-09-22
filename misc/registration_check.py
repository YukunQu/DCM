import os
import numpy as np
import pandas as pd
from nilearn.image import load_img,math_img,mean_img,new_img_like


# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('game1_fmri==1').copy()  # look out
subjects = data['Participant_ID']

# load func image for runs of each subject
fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game1/separate_hexagon/Setall/6fold/work_1st'
sub_img = []

all_sub_beta_sum = []
for sub in subjects:
    sub = sub.split("-")[-1]
    if int(sub)==151:
        continue
    beta_dir = os.path.join(fmriprep_dir,f'_subj_id_{sub}','level1estimate')
    file_list = os.listdir(beta_dir)
    beta_list = [load_img(os.path.join(beta_dir,f)).get_fdata() for f in file_list if 'beta' in f]
    sub_beta_sum = np.sum(np.array(beta_list),axis=0)
    print(sub,"Nan number", np.sum(~np.isfinite(sub_beta_sum)))
    all_sub_beta_sum.append(sub_beta_sum)
all_sub_beta_sum = np.sum(np.array(all_sub_beta_sum),axis=0)


template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game1/separate_hexagon/Setall/6fold/work_1st/_subj_id_078/level1estimate/beta_0001.nii'
subs_beta_map = new_img_like(template,data=all_sub_beta_sum)
subs_beta_map.to_filename(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game1/separate_hexagon/Setall/6fold/all_sub_beta_map.nii.gz')
