import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img
#%%
# set subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('usable==1').query("Age>18")
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

# set target roi
roi_name = 'EC'

folds = [str(i) + 'fold' for i in range(4, 9)]
rois = {'EC': r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/defROI/adult/EC_func_roi.nii',
        'vmPFC': r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/defROI/adult/vmpfc_func_roi.nii'}
#%%
# pipeline
subjects_data = pd.DataFrame(columns=['ifold', 'sub_id', 'testset','amplitude'])
for ifold in folds:
    print(f"________{ifold} start____________")
    for sub in subjects:
        for set_id in [1,2]:
            stats_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/specificTo6/test_set/ec/' \
                       f'testset{set_id}/{ifold}/{sub}/spmT_0001.nii'
            roi_img = load_img(rois[roi_name])
            amplitude = np.nanmean(apply_mask(imgs=stats_map, mask_img=roi_img))
            sub_data = {'sub_id': sub, 'ifold': ifold, 'testset': str(set_id),'amplitude': amplitude}
            subjects_data = subjects_data.append(sub_data, ignore_index=True)

subjects_data.to_csv(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/specificTo6/test_set/ec/'  #look out 
                          r'subjects_ifold_{}_testPhi_adults.csv'.format(roi_name), index=False)