import json
import os

import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img,resample_to_img

# set path template:
analysis_type = 'alignPhiGame1'

stats_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/alignPhiGame1/specificTo6/test_set/EC_individual/testset{}/{}/{}/Mcon_0001.nii'
roi_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/defROI/EC/{}_EC_func_roi.nii'
save_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/betas/alignPhi_EC_betas_individual_ROI_all.csv'

# set test set
testset = [1, 2]

# set subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
#data = data.query('game1_acc>=0.8')
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

subjects = os.listdir(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/alignPhiGame1/specificTo6/test_set/EC_individual/testset2/6fold')

folds = range(4,9)

sub_fold_beta = pd.DataFrame(columns=['sub_id','ifold','set_id','amplitude'])
for i in folds:
    ifold = str(i)+'fold'
    print(f"________{ifold} start____________")
    for sub in subjects:
        for set_id in testset:
            stats_map = stats_path.format(set_id,ifold,sub)
            roi_img = load_img(roi_path.format(sub))
            #roi_img = load_img(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/defROI/EC/group_EC_func_roi.nii')
            roi_img = resample_to_img(roi_img, stats_map,interpolation='nearest')
            amplitude = np.nanmean(apply_mask(imgs=stats_map, mask_img=roi_img))
            tmp_data = {'sub_id': sub, 'ifold': ifold, 'set_id': str(set_id),'amplitude': amplitude}
            sub_fold_beta = sub_fold_beta.append(tmp_data, ignore_index=True)
sub_fold_beta.to_csv(save_path,index=False)