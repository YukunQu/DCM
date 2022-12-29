import os
from os.path import join as pjoin
import pandas as pd
from analysis.mri.voxel_wise.secondLevel import level2nd_covar_age,level2nd_covar_acc,level2nd_onesample_ttest,level2nd_covar_acc_age

# --------------------------Set configure --------------------------------
# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
#data = data.query("Participant_ID!='sub-186'")

#contrast_1st = ['ZF_0005']
#contrast_1st = ['con_0001','con_0002','con_0003','con_0004','ZF_0005']
contrast_1st = ['con_0001','con_0002','con_0003','con_0004','con_0005']
#contrast_1st = ['ZF_0005','ZF_0006','ZF_0011','con_0007','con_0008','con_0012','con_0013','con_0014']
#contrast_1st = ['con_0001','con_0002','ZF_0003']
#contrast_1st = ['rs-corr_img_coarse','rs-corr_zmap_coarse']

data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
task = 'game1'
glm_type = 'cv_test1_bigmPFC_weighted-average'
set_id = 'Setall'
ifold = '6fold'
templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}','sub-{subj_id}', '{contrast_id}.nii')}

# ----------------Mean effect for all subjects -----------------------------
# set subjects
pid = data['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]
out_container = f'{task}/{glm_type}/{set_id}/{ifold}/group/all'  # output = data_root + out_container
level2nd_onesample_ttest(sub_list,contrast_1st,data_root,templates,out_container)

# ------------------Mean effect for high performance adults--------------------
# set subjects
hp_data = data.query("(game1_acc>=0.80)and(Age>=18)")  # look out
hp_pid = hp_data['Participant_ID'].to_list()
hp_sub_list = [p.split('-')[-1] for p in hp_pid]
out_container = f'{task}/{glm_type}/{set_id}/{ifold}/group/hp-adult'  # output = data_root + out_container
level2nd_onesample_ttest(hp_sub_list,contrast_1st,data_root,templates,out_container)

hp_data = data.query("(game1_acc>=0.7)and(Age>=13)")  # look out
hp_pid = hp_data['Participant_ID'].to_list()
hp_sub_list = [p.split('-')[-1] for p in hp_pid]
out_container = f'{task}/{glm_type}/{set_id}/{ifold}/group/acc_0.7_age_13'  # output = data_root + out_container
print(len(hp_sub_list))
level2nd_onesample_ttest(hp_sub_list,contrast_1st,data_root,templates,out_container)

# -----------------Covariate analysis-------------------------------------------
level2nd_covar_acc(data,task,glm_type,set_id,ifold,contrast_1st)
level2nd_covar_age(data,task,glm_type,set_id,ifold,contrast_1st)
#level2nd_covar_acc_age(data,task,glm_type,set_id,ifold,contrast_1st)