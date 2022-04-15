import json

import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img,resample_to_img

# set path template:

stats_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/test_set/EC/testset{}/{}/{}/Mcon_0001.nii'
roi_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/defROI/EC/{}_EC_func_roi.nii'
save_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/betas/alignPhi_EC_betas_individual_ROI.csv'

# set test set
testset = [1, 2]

# set subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('(usable==1)&(game1_acc>0.75)&(Age>18)')
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

folds = range(4,9)

sub_fold_beta = pd.DataFrame(columns=['sub_id','ifold','set_id','amplitude'])
for i in folds:
    ifold = str(i)+'fold'
    print(f"________{ifold} start____________")
    for sub in subjects:
        for set_id in testset:
            stats_map = stats_path.format(set_id,ifold,sub)
            roi_img = load_img(roi_path.format(sub))
            #roi_img = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/'
            #                   'osfstorage-archive/data/Analysis_ROI_nii/EC_Grid_roi.nii')
            roi_img = resample_to_img(roi_img, stats_map,interpolation='nearest')
            amplitude = np.nanmean(apply_mask(imgs=stats_map, mask_img=roi_img))
            tmp_data = {'sub_id': sub, 'ifold': ifold, 'set_id': str(set_id),'amplitude': amplitude}
            sub_fold_beta = sub_fold_beta.append(tmp_data, ignore_index=True)
sub_fold_beta.to_csv(save_path,index=False)