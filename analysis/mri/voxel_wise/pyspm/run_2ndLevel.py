from os.path import join as pjoin
import pandas as pd
from analysis.mri.voxel_wise.pyspm.secondLevel import level2nd_covar_age, level2nd_covar_acc, level2nd_onesample_ttest

# --------------------------Set configure --------------------------------

# subject
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
sub_info = participants_data.query('game2_fmri>=0.5')  # look out
print("{} subjects".format(len(sub_info)))

# parameters
configs = {'data_root': r'/mnt/data/DCM/derivatives/Nipype',
           'task': 'game2',
           'glm_type': 'distance_spct',
           'set_id': 'Setall',
           'ifold': '6fold'}

contrast_set = {'distance_spat': ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005'],
                'distance_spct': ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006'],
                'hexagon_spat': ['con_0001', 'con_0002', 'con_0003', 'con_0004',
                                 'con_0007', 'con_0008', 'con_0009', 'con_0010',
                                 'zstats_0005', 'zstats_0006', 'zstats_0011'],
                'hexagon_spct': ['con_0001', 'con_0002', 'con_0003', 'con_0004',
                                 'con_0007', 'con_0009', 'con_0010','con_0013', 'con_0015',
                                 'zstats_0005', 'zstats_0006', 'zstats_0011']}

contrast_1st = contrast_set[configs['glm_type']]

# ----------------Mean effect for all subjects -----------------------------
# set subjects
pid = sub_info['Participant_ID'].to_list()
sub_list = [p.split('-')[-1] for p in pid]
level2nd_onesample_ttest(sub_list, contrast_1st, configs)

# -----------------Covariate analysis-------------------------------------------
level2nd_covar_acc(sub_info, contrast_1st, configs)
level2nd_covar_age(sub_info, contrast_1st, configs)