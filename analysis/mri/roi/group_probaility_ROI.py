import numpy as np
import pandas as pd
from nilearn.image import get_data,new_img_like,load_img

#%%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'game1_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]
roi_template = r'/mnt/workdir/DCM/result/ROI/individual/{}_roi.nii.gz'

roi_example = roi_template.format(subject_list[0])
data = get_data(roi_example)
roi_sum = np.zeros(data.shape)
for i,sub in enumerate(subject_list):
    roipath = roi_template.format(sub)
    roi_tmp = load_img(roipath)
    roi_data = get_data(roipath)
    roi_sum += roi_data

roi_sum_img = new_img_like(roi_example, roi_sum)
roi_sum_img.to_filename(r'/mnt/workdir/DCM/result/ROI/individual/roi_sum.nii.gz')

#%%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'game1_fmri>=0.5')  # look out
data = data.query("(game1_acc>=0.80)and(Age>=18)")
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]

# get all path of stats_map
stats_map_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/m2_hexagon_correct_trials/Setall/6fold/{}/ZF_0005.nii'
stats_example = stats_map_template.format(subject_list[0])
data = get_data(stats_example)
roi_sum = np.zeros(data.shape)

for i,sub in enumerate(subject_list):
    stats_map_path = stats_map_template.format(sub)
    stats_tmp = load_img(stats_map_path)
    stats_data = get_data(stats_tmp)
    stats_data[stats_data<1.68] = 0
    stats_data[stats_data>=1.68] = 1
    roi_sum += stats_data

roi_sum = roi_sum/len(subject_list)
roi_sum_img = new_img_like(stats_example, roi_sum)
check_img = new_img_like(stats_example,stats_data)
roi_sum_img.to_filename(r'/mnt/workdir/DCM/result/ROI/individual/roi_map_hp_thr1.68.nii.gz')
check_img.to_filename(r'/mnt/workdir/DCM/result/ROI/individual/check_img.nii.gz')
