import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('usable==1').query("Age>18")
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

folds = [str(i)+'fold' for i in range(4,9)]
rois =  {'EC': r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/defROI/adult/EC_func_roi.nii',
         'vmPFC': r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/defROI/adult/vmpfc_func_roi.nii'}
subjects_beta_data = pd.DataFrame(columns=['sub_id','ifold','testset','EC_beta','vmPFC_beta'])

for sub in subjects:
    print(f"________{sub} start____________")
    for ifold in folds:
        for set_id in [1,2]:
            beta_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/specificTo6/test_set/' \
                       f'testset{set_id}/{ifold}/{sub}/con_0001.nii'

            ec_roi = load_img(rois['EC'])
            ec_beta = np.nanmean(apply_mask(imgs=beta_map, mask_img=ec_roi))

            vmpfc_roi = load_img(rois['vmPFC'])
            vmpfc_beta = np.nanmean(apply_mask(imgs=beta_map, mask_img=vmpfc_roi))

            sub_beta_data = {'sub_id':sub, 'ifold': ifold,'testset': str(set_id),
                                          'EC_beta':ec_beta, 'vmPFC_beta':vmpfc_beta}
            subjects_beta_data = subjects_beta_data.append(sub_beta_data,ignore_index=True)

subjects_beta_data.to_csv(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/specificTo6/'
                          r'subjects_ifold_beta_data_adults.csv',index=False)