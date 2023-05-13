# plot the distribution of mean orientation for all subjects
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-pastel")

# read high performance subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('(game1_fmri>=0.5)')
subjects = data['Participant_ID'].to_list()

# read the mean orientation of all subjects
phis_file = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_spct/estPhi_ROI-EC_circmean_trial-all.csv'
phis_data = pd.read_csv(phis_file)
phis_data = phis_data.query("(sub_id in @subjects)and(ifold=='6fold')")

# extract the "Phi_mean" column
phi_mean = phis_data['Phi_mean']

# plot the angle distribution
plt.hist(phi_mean, bins=50, range=(-5, 65))
plt.xlabel('Phi_mean')
plt.ylabel('Count')
plt.title('Angle Distribution for Phi_mean')
plt.show()
