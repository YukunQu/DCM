import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm import threshold_stats_img


def run_2nd_ttest(subjects, contrast_id, cmap_template, datasink):
    # Select cmaps
    cmaps = [cmap_template.format(sub_id, contrast_id) for sub_id in subjects]
    # Set design matrix
    design_matrix = pd.DataFrame([1] * len(cmaps), columns=['intercept'])
    # run glm
    glm_2nd = SecondLevelModel(smoothing_fwhm=6.0)
    glm_2nd = glm_2nd.fit(cmaps, design_matrix=design_matrix)
    stats_map = glm_2nd.compute_contrast(second_level_contrast='intercept',output_type='all')
    t_map = stats_map['stat']
    z_map = stats_map['z_score']
    # save
    if not os.path.exists(datasink):
        os.makedirs(datasink)

    t_image_path = os.path.join(datasink,'%s_tmap.nii.gz' % contrast_id)
    t_map.to_filename(t_image_path)

    z_image_path = os.path.join(datasink, '%s_zmap.nii.gz' % contrast_id)
    #z_map.to_filename(z_image_path)


def run_2nd_covariate(subjects, contrast_id, cmap_template, covariates, datasink):
    # Select cmaps
    cmaps = [cmap_template.format(sub_id, contrast_id) for sub_id in subjects]

    # Set design matrix
    covariate_name = covariates.keys()
    extra_info_subjects = pd.DataFrame(covariates)
    extra_info_subjects['subject_label'] = subjects

    design_matrix = make_second_level_design_matrix(subjects, extra_info_subjects)
    for key in covariate_name:
        design_matrix[key] = design_matrix[key] - design_matrix[key].mean()

    # estimate second level model
    glm_2nd = SecondLevelModel(smoothing_fwhm=6.0)
    glm_2nd = glm_2nd.fit(cmaps, design_matrix=design_matrix)

    if not os.path.exists(datasink):
        os.mkdir(datasink)

    for index, covariate in enumerate(covariate_name):
        stats_map = glm_2nd.compute_contrast(covariate, output_type='all')
        t_map = stats_map['stat']
        z_map = stats_map['z_score']

        # write the resulting stat images to file
        tmap_path = pjoin(datasink, '{}_{}_tmap.nii.gz'.format(contrast_id, covariate))
        t_map.to_filename(tmap_path)

        zmap_path = pjoin(datasink, '{}_{}_zmap.nii.gz'.format(contrast_id, covariate))
        #z_map.to_filename(zmap_path)


if __name__ == "__main__":
    # subject
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    pid = data['Participant_ID'].to_list()
    sub_list = [p.split('-')[-1] for p in pid]

    print("{} subjects".format(len(sub_list)))

    hp_data = data.query("(game1_acc>=0.80)and(Age>=18)")  # look out
    hp_pid = hp_data['Participant_ID'].to_list()
    hp_sub_list = [p.split('-')[-1] for p in hp_pid]
    print("{} hp subjects".format(len(hp_sub_list)))

    # configure
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nilearn_test'
    task = 'game1'
    glm_type = 'distance_whole_trials'
    set_id = 'Setall'
    ifold = '6fold'
    templates = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}','sub-{}/zmap', '{}_zmap.nii.gz')

    contrast_1st = ['M1', 'M2', 'decision','M2xdistance','decisionxdistance','distance']

    data_root = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}','group')
    for contrast_id in contrast_1st:
        datasink = os.path.join(data_root,'mean')
        run_2nd_ttest(sub_list, contrast_id, templates, datasink)

        datasink = os.path.join(data_root,'hp')
        run_2nd_ttest(hp_sub_list, contrast_id, templates, datasink)

        # age
        age = data['Age'].to_list()
        covariates = {'Age':age}
        datasink = os.path.join(data_root,'age')
        run_2nd_covariate(sub_list, contrast_id, templates, covariates, datasink)

        # acc
        accuracy = data['game1_acc'].to_list()
        covariates = {'acc':accuracy}
        datasink = os.path.join(data_root,'acc')
        run_2nd_covariate(sub_list, contrast_id, templates, covariates, datasink)