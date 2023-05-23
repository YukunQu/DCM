import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# calculate phis correlation between different GLM

# load participant
participants = pd.read_csv(r"/mnt/workdir/DCM/BIDS/participants.tsv", sep='\t')
sub_id = participants.query('game1_fmri>0')['Participant_ID'].tolist()


# load data
phis_all = pd.read_csv(r"/mnt/data/DCM/result_backup/2023.3.24/Nilearn_smodel/game1/hexagon_spct/estPhi_ROI-EC_circmean_trial-all.csv")
phis_all = phis_all.query("ifold=='6fold'")
phis_all = phis_all[phis_all['sub_id'].isin(sub_id)]
phis_mean = phis_all['Phi_mean']

phis1 = pd.read_csv(r"/mnt/data/DCM/result_backup/2023.3.24/Nilearn_smodel/game1/"
                    r"cv_train_hexagon_spct/estPhi_ROI-EC_circmean_cv.csv")
phis1 = phis1.query("ifold=='6fold'")
phis1 = phis1[phis1['sub_id'].isin(sub_id)]
phis1_odd = phis1[phis1['trial_type'] == 'odd']['Phi_mean']
phis1_even = phis1[phis1['trial_type'] == 'even']['Phi_mean']

phis2 = pd.read_csv(r"/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/cv_train_hexagon_spct/estPhi_ROI-EC-anat_circmean_cv.csv")
phis2_odd = phis2[phis2['trial_type'] == 'odd']['Phi_mean']
phis2_even = phis2[phis2['trial_type'] == 'even']['Phi_mean']

# calculate correlation
corr_odd,p_odd = pearsonr(phis_mean, phis2_odd)
corr_even,p_even = pearsonr(phis_mean, phis2_even)

# print results
print("Correlation between odd trials: ", round(corr_odd,4),round(p_odd,5))
print("Correlation between even trials: ", round(corr_even,4),round(p_even,5))