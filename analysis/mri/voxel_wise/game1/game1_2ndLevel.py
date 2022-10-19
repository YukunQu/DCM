import os

import pandas as pd
from analysis.mri.voxel_wise.secondLevel import level2nd_covar_age,level2nd_covar_acc,level2nd_noPhi,level2nd_covar_acc_age

# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
#%%
task = 'game1'  # look out
glm_type = 'separate_hexagon_old'
contrast_1st = ['ZF_0005','ZF_0006','ZF_0011','ZT_0007','ZT_0008']

level2nd_covar_age(data,task,glm_type, contrast_1st)
level2nd_covar_acc(data,task,glm_type, contrast_1st)

# all subject mean effect
pid = data['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]
level2nd_noPhi(sub_list,'all',task,glm_type, 'Setall',contrast_1st)

# high performance subject mean effect
hp_data = data.query("(game1_acc>=0.8)and(Age>=18)")
hp_pid = hp_data['Participant_ID'].to_list()
hp_sub_list = [p.split('-')[-1] for p in hp_pid]
level2nd_noPhi(hp_sub_list,'hp',task,glm_type, 'Setall', contrast_1st)

#  mean effect of adults
adult_data = data.query("Age>=18")
adult_pid = adult_data['Participant_ID'].to_list()
adult_sub_list = [p.split('-')[-1] for p in adult_pid]
level2nd_noPhi(adult_sub_list,'adult',task,glm_type, 'Setall', contrast_1st)