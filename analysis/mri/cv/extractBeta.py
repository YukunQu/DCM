import os
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.image import load_img, resample_to_img,threshold_img,binarize_img


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
                stats_map = stat_map.format(ifold,sub)
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
    roi_path = r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz'
    roi_img = load_img(roi_path)
    roi_img = binarize_img(roi_img,30)
    #roi_img = load_img(r'/mnt/workdir/DCM/Docs/Mask/Park_Grid_ROI/EC_Grid_roi.nii')
    #roi_img = load_img(r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/hexagon_spct/EC_thr3.1.nii.gz')

    # set path template:
    stats_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_test_hexagon_spct/Setall/{}/{}/zmap/alignPhi_even_zmap.nii.gz'
    #stats_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/grid_rsa_corr_trials/Setall/6fold/{}/rsa/rsa_img_coarse_{}.nii.gz'

    # set savepath
    savedir = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_test_hexagon_spct'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    save_path = os.path.join(savedir,'sub_stats-z_roi-ec_trial-even_anat_EC_thr20.csv')

    sub_stats_results = extractStats(stats_template,subjects,roi_img)
    sub_stats_results['trial_type'] = 'even'
    #sub_stats_results.to_csv(save_path)

    # high performance filter
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    hp_info = participants_data.query(f'(game1_fmri>=0.5)and(game1_acc>0.8)')  # look out
    hp_sub = hp_info['Participant_ID'].to_list()
    data = sub_stats_results.loc[sub_stats_results['sub_id'].isin(hp_sub)]

    from scipy.stats import ttest_1samp
    for i in range(4,9):
        ifold = str(i)+'fold'
        fold6_act = data[data['ifold']==ifold]['statistic'].to_list()
        _,p = ttest_1samp(fold6_act,0)
        p = round(p,5)
        print('one sample t-test for {}fold: pvalue={}'.format(i,str(p).zfill(3)))