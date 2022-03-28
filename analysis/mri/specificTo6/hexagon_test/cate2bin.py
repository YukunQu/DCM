import pandas as pd
from os.path import join as pjoin


def cat2bin(activities, ifold, phi):
    # categorize M2 bold activity into k bin.
    # num of bin == periodicity * 2
    test_angles = activities['angle']
    alignedD_360 = [(a - phi) % 360 for a in test_angles]

    periodicity = 360 / ifold
    bin_num = int(360 / (periodicity / 2))
    anglebinID = [round(a / (periodicity / 2)) + 1 for a in alignedD_360]
    anglebinID = [1 if b == (bin_num + 1) else b for b in anglebinID]

    label = []
    for bin_id in anglebinID:
        if bin_id in range(1, bin_num + 1, 2):
            label.append('align')
        elif bin_id in range(2, bin_num + 1, 2):
            label.append('misalign')

    activities['bin'] = anglebinID
    activities['label'] = label
    return activities


# %%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('usable==1').query('Age>18')
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_', '-') for p in pid]

ifold = 6
activity_data_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/brain_activity'

sub_bin_mean_act = pd.DataFrame()
sub_label_mean_act = pd.DataFrame()
for sub in subjects:
    # read phi of subject based on ifold
    phi_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexonM2Long/specificTo6/Phi/backup'
    phi_data = pd.read_csv(pjoin(phi_dir, 'trainsetall_estPhi_mean.csv'))
    phi = phi_data.query('(sub_id=="{}")and(ifold=="{}")'.format(sub, str(ifold) + 'fold'))['ec_phi'].values[0]

    # read subject's activities
    sub_activity = pd.read_csv(pjoin(activity_data_dir, sub, '{}_EC_M2_activity.csv'.format(sub)))
    sub_activity = sub_activity

    # category mean activity into bins
    sub_activity_bins = cat2bin(sub_activity, ifold, phi)
    bin_mean_act = sub_activity_bins.groupby('bin').mean()['bold']
    label_mean_act = sub_activity_bins.groupby('label').mean()['bold']

    sub_bin_mean_act = sub_bin_mean_act.append(bin_mean_act)
    sub_label_mean_act = sub_label_mean_act.append(label_mean_act)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

columns = sub_bin_mean_act.columns
x = []
y = []
for bin in columns:
    y.extend(sub_bin_mean_act[bin].to_list())
    x.extend((len(sub_bin_mean_act)) * [str(bin)])
data_plot = pd.DataFrame({'BinNumber': x,
                          'Activation': y})

label = []
for bin_id in x:
    bin_id = int(bin_id)
    if bin_id in range(1, 12 + 1, 2):
        label.append('align')
    elif bin_id in range(2, 12 + 1, 2):
        label.append('misalign')
data_plot['label'] = label
sns.catplot(data=data_plot, kind='bar', x='BinNumber',
            y='Activation', hue='label')
# plt.bar(x=x,height=y)
plt.tight_layout()
plt.show()
# %%
sns.catplot(data=data_plot, kind='bar', x='label',
            y='Activation')
# plt.bar(x=x,height=y)
plt.tight_layout()
plt.show()
