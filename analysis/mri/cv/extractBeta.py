import os

import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img, resample_to_img


def extractStats(stat_map,subjects,roi):
    """ extract mean statistic of ROI from statistical map
    extract
    :param stat_map: the template of statistical map
                    eg:
    :param subjects: subject_list
    :param roi: target_roi
    :return:
    """
    folds = range(4, 9)
    sub_stats = pd.DataFrame(columns=['ifold','sub_id', 'amplitude'])

    for i in folds:
        ifold = str(i) + 'fold'
        print(f"________{ifold} start____________")
        for sub in subjects:
                stats_map = stat_map.format(ifold, sub)
                print(stats_map)
                stats_img = load_img(stats_map)
                if not np.array_equal(stats_img.affine,roi.affine):
                    raise Exception("The mask havs a different affine matrix to subject's statistical map.")
                stats = apply_mask(imgs=stats_img, mask_img=roi)
                stats[stats == 0] = np.nan
                amplitude = np.nanmean(stats)
                tmp_data = {'ifold': ifold, 'sub_id': sub, 'statistic': amplitude,'Age':'','Acc':''}
                sub_stats = sub_stats.append(tmp_data, ignore_index=True)
    return sub_stats


if __name__ == "__main__":
    # set subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')
    subjects = data['Participant_ID'].to_list()

    # set roi
    roi_path = r'/mnt/workdir/DCM/result/ROI/Ftest/EC_thr3.7_anat50.nii.gz'
    roi_img = load_img(roi_path)

    # set path template:
    stats_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/cv_test_hexagon_distance_spct_ECthr3.1/' \
                     r'Setall/{}/{}/zmap/alignPhi_zmap.nii.gz'

    # set savepath
    savedir = r'/mnt/workdir/DCM/result/Specificity_to_6/nilearn_cv'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    save_path = r'/mnt/workdir/DCM/result/Specificity_to_6/nilearn_cv/sub_stats-z_roi-ec3.7_trial-all.csv'

    sub_stats_results = extractStats(stats_template,subjects,roi_img)
    sub_stats_results['trial_type'] = 'all'
    sub_stats_results.to_csv(save_path)

    """
    for sub in subjects:
        age = data.loc[data.Participant_ID == sub_tmp, 'Age'].values[0]
        acc = data.loc[data.Participant_ID == sub_tmp, 'game1_acc'].values[0]
        tmp_data = {'sub_id': sub, 'amplitude': amplitude, 'age': age, 'acc': acc}
        sub_fold_beta = sub_fold_beta.append(tmp_data, ignore_index=True)
    sub_fold_beta.to_csv(save_path, index=False)
    """
