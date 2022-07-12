import pandas as pd
from analysis.mri.wba.secondLevel import level2nd_covar_age,level2nd_covar_acc,level2nd_noPhi

# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]

task = 'game1'  # look out
glm_type = 'separate_hexagon'

contrast_1st = ['ZF_0005','ZF_0006','ZT_0007','ZT_0008','ZF_0011','ZT_0012']

#level2nd_noPhi(subject_list,'all',task,glm_type, 'Setall',contrast_1st)
level2nd_covar_age(data,task,glm_type,contrast_1st)
level2nd_covar_acc(data,task,glm_type,contrast_1st)