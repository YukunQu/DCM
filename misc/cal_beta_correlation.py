import pandas as pd
from nilearn.masking import apply_mask
from scipy.stats import pearsonr

# subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
#data = data.query("(game1_acc>=0.7)and(Age>=13)")
sub_list = data['Participant_ID'].to_list()

# set roi
roi_odd = r'/mnt/workdir/DCM/result/ROI/Group/F-test_mPFC_thr2.3.nii.gz'
roi_even = r'/mnt/workdir/DCM/result/ROI/Group/F-test_mPFC_thr2.3.nii.gz'

# set stats_map
sin_odd_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/cv_train1/Setall/6fold/{}/con_0002.nii'
sin_even_map = '/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/cv_train1/Setall/6fold/{}/con_0004.nii'
sin_odd_beta = []
sin_even_beta = []

for sub_id in sub_list:
    sin_odd_beta.append(apply_mask(sin_odd_map.format(sub_id),roi_odd).mean())
    sin_even_beta.append(apply_mask(sin_even_map.format(sub_id),roi_even).mean())

r,p = pearsonr(sin_odd_beta,sin_even_beta)
print('sin:','r:',round(r,3),'p:',round(p,3))


cos_odd_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/cv_train1/Setall/6fold/{}/con_0001.nii'
cos_even_map = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/cv_train1/Setall/6fold/{}/con_0003.nii'
cos_odd_beta = []
cos_even_beta = []
for sub_id in sub_list:
    cos_odd_beta.append(apply_mask(cos_odd_map.format(sub_id),roi_odd).mean())
    cos_even_beta.append(apply_mask(cos_even_map.format(sub_id),roi_even).mean())

r,p = pearsonr(cos_odd_beta,cos_even_beta)
print('cos:','r:',round(r,3),'p:',round(p,3))