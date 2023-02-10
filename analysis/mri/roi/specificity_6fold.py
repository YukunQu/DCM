import numpy as np
import pandas as pd
from nilearn.image import load_img, get_data, threshold_img, binarize_img
from nilearn.masking import apply_mask


def extract_roi_mean(subs_stats_map, roi):
    stats_map = load_img(subs_stats_map[0])
    mask = load_img(roi)
    if not np.array_equal(stats_map.affine, mask.affine):
        print(stats_map.affine)
        print(mask.affine)
        raise Exception("The affines of data and roi are not the same.")
    values = apply_mask(subs_stats_map, mask)
    mean_amplitude = np.nanmean(values, axis=1)
    return mean_amplitude


# def extract_subs_roi_mean():
#  set subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
data = data.query("(game1_acc>=0.80)and(Age>=18)")
subjects = data['Participant_ID'].to_list()

#  set cmap tempalte
stats_map_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_corr_trials/Setall/6fold/{}/rs-corr_ztransf_map_coarse_{}fold.nii'

# set ROI
roi = load_img(r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL_prob.nii.gz')
# roi = load_img(r'/mnt/workdir/DCM/docs/Mask/Park_Grid_ROI/EC_Grid_roi.nii')
roi_thr_bin = binarize_img(roi, 20)
# roi_thr_bin.to_filename(r'/mnt/workdir/DCM/result/ROI/anat/juelich_EC_MNI152NL_prob_L_thr99.nii.gz')

# set fold
folds = range(4, 9)

sub_roi_amp = pd.DataFrame(columns=['sub_id', 'ifold', 'amplitude'])
for ifold in folds:
    print(f"________{ifold}fold start____________")
    subs_stats_map = [stats_map_tempalte.format(s, ifold) for s in subjects]
    amplitude = extract_roi_mean(subs_stats_map, roi_thr_bin)
    tmp_dict = pd.DataFrame({'sub_id': subjects, 'ifold': [ifold] * len(subjects), 'amplitude': amplitude})
    sub_roi_amp = sub_roi_amp.append(tmp_dict, ignore_index=True)
# sub_roi_amp.to_csv(r'/mnt/workdir/DCM/result/Specificity_to_6/RSA/sub-hp_ROI-EClthr99_amplitude-z.csv', index=False)

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_style("darkgrid")

# sub_roi_amp = pd.read_csv( r'/mnt/workdir/DCM/result/Specificity_to_6/RSA/sub-hp_ROI-EClthr95_amplitude-z.csv')

fig, ax = plt.subplots()
sns.boxplot(x='ifold', y="amplitude", data=sub_roi_amp, width=.2,
            palette=["lightgray", "lightgray", "steelblue", "lightgray", "lightgray"],
            boxprops={'edgecolor': 'None'},
            )
sns.stripplot(x='ifold', y="amplitude", data=sub_roi_amp, size=2, color='.3', linewidth=0)
x = [0, 1, 2, 3, 4]
y = [0] * len(x)
plt.plot(x, y, linestyle='--', color='gray')
plt.xticks(size=16)
plt.xlabel('fold', size=16)
plt.ylabel('Zscore')
sub_num = len(set(sub_roi_amp['sub_id']))
plt.title("High performance participants(num={})".format(sub_num), size=16)
plt.show()

# %%
from scipy.stats import ttest_1samp

ifold_p = []
for i in range(4, 9):
    fold6_act = sub_roi_amp.query(f'ifold=={i}')['amplitude'].to_list()
    _, p = ttest_1samp(fold6_act, 0, alternative='greater')
    ifold_p.append(p)
    p = round(p, 6)
    print('one sample t-test result of {}fold: pvalue={}'.format(i, p))

# %%

from scipy.stats import ttest_rel

act1 = sub_roi_amp.query(f'ifold==6')['amplitude'].to_list()
act2 = sub_roi_amp.query(f'ifold==8')['amplitude'].to_list()
_, p = ttest_rel(act2, act1)
p = round(p, 6)
print('pair t-test result: pvalue={}'.format(p))
