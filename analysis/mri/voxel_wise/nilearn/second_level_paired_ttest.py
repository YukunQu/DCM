import numpy as np
import pandas as pd
from os.path import join as pjoin
from nilearn.glm.second_level import SecondLevelModel
from nilearn.datasets import fetch_localizer_contrasts
from joblib import Parallel, delayed


def run_2nd_paired_ttest(subjects,contrast_id,cmap1_temp,cmap2_temp,datasink):
    # Select cmaps
    cmap1s = [cmap1_temp.format(sub_id, contrast_id) for sub_id in subjects]
    cmap2s = [cmap2_temp.format(sub_id, contrast_id) for sub_id in subjects]
    # define the input maps
    second_level_input = cmap1s + cmap2s

    # model the effect of condition
    n_subjects = len(sub_list)
    condition_effect = np.hstack(([-1] * n_subjects, [1] * n_subjects))

    subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
    subjects = [f'S{i:03d}' for i in range(1, n_subjects + 1)]

    paired_design_matrix = pd.DataFrame(np.hstack((condition_effect[:, np.newaxis], subject_effect)),
                                        columns=['game1 vs game2'] + subjects)

    # second level analysis
    second_level_model_paired = SecondLevelModel().fit(
        second_level_input, design_matrix=paired_design_matrix)
    stat_maps_paired = second_level_model_paired.compute_contrast('game1 vs game2',output_type='all')
    c_map = stat_maps_paired['effect_size']
    t_map = stat_maps_paired['stat']
    z_map = stat_maps_paired['z_score']

    # write the resulting stat images to file
    cmap_path = pjoin(datasink, '{}_tmap.nii.gz'.format(contrast_id))
    c_map.to_filename(cmap_path)

    tmap_path = pjoin(datasink, '{}_tmap.nii.gz'.format(contrast_id))
    t_map.to_filename(tmap_path)

    zmap_path = pjoin(datasink,'{}_zmap.nii.gz'.format(contrast_id))
    z_map.to_filename(zmap_path)


if __name__ == "__main__":
    task = 'game2'
    # subject
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')  # look out
    pid = data['Participant_ID'].to_list()
    sub_list = [p.split('-')[-1] for p in pid]

    contrast_1st = ['correct_error','M2_corrxdistance','decision_corrxdistance']

    game1_cmap_temp = r'/mnt/data/DCM/result_backup/2023.3.24/Nilearn_smodel/game1/distance_spct/Setall/6fold/sub-{}/zmap/{}_zmap.nii.gz'
    game2_cmap_temp = r'/mnt/data/DCM/result_backup/2023.3.24/Nilearn_smodel/game2/distance_spct/Setall/6fold/sub-{}/zmap/{}_zmap.nii.gz'

    dataout = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/compare_2game/paired_ttest'

    for contrast in contrast_1st:
        run_2nd_paired_ttest(sub_list,contrast,game1_cmap_temp,game2_cmap_temp,dataout)