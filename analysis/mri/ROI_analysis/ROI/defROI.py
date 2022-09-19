import os
import pandas as pd
from analysis.mri.ROI_analysis.ROI.makeMask import makeSphereMask

task = 'game1'
glm_type = 'separate_hexagon'

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'{task}_fmri==1')  # look out
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]

savedir = '/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/defROI/EC/individual'.format(task)
if not os.path.exists(savedir):
    os.makedirs(savedir)

for sub in subject_list:
    stats_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/Setall/6fold/sub-{sub}/ZF_0011.nii'
    roi = r'/mnt/workdir/DCM/docs/Reference/EC_ROI/volume/EC-thr50-2mm.nii.gz'
    savepath = os.path.join(savedir,'{}_EC_func_roi.nii'.format(sub))
    makeSphereMask(stats_map, roi, savepath, radius=(5/3,5/3,5/3))

savedir = '/mnt/workdir/DCM/BIDS/derivatives/Nipype/{}/defROI/vmpfc/individual'.format(task)
if not os.path.exists(savedir):
    os.makedirs(savedir)

for sub in subject_list:
    stats_map = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{task}/{glm_type}/Setall/6fold/sub-{sub}/ZF_0011.nii'
    roi = r'/mnt/data/Template/VMPFC_roi.nii'
    savepath = os.path.join(savedir,'{}_vmpfc_func_roi.nii'.format(sub))
    makeSphereMask(stats_map, roi, savepath, radius=(5/3,5/3,5/3))