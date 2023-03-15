import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm import threshold_stats_img
from joblib import Parallel, delayed


def run_2nd_ttest(subjects, contrast_id, cmap_template, datasink):
    # Select cmaps
    cmaps = [cmap_template.format(sub_id, contrast_id) for sub_id in subjects]
    # Set design matrix
    design_matrix = pd.DataFrame([1] * len(cmaps), columns=['intercept'])
    # run glm
    glm_2nd = SecondLevelModel(smoothing_fwhm=8.0)
    glm_2nd = glm_2nd.fit(cmaps, design_matrix=design_matrix)
    stats_map = glm_2nd.compute_contrast(second_level_contrast='intercept', output_type='all')
    t_map = stats_map['stat']
    z_map = stats_map['z_score']

    # write the resulting stat images to file
    t_image_path = os.path.join(datasink, '%s_tmap.nii.gz' % contrast_id)
    t_map.to_filename(t_image_path)

    z_image_path = os.path.join(datasink, '%s_zmap.nii.gz' % contrast_id)
    z_map.to_filename(z_image_path)


def run_2nd_covariate(subjects, contrast_id, cmap_template, covariates, datasink):
    # Select cmaps
    cmaps = [cmap_template.format(sub_id, contrast_id) for sub_id in subjects]

    # Set design matrix
    extra_info_subjects = pd.DataFrame(covariates)
    extra_info_subjects['subject_label'] = subjects
    design_matrix = make_second_level_design_matrix(subjects, extra_info_subjects)
    covariate_name = covariates.keys()
    for key in covariate_name:
        design_matrix[key] = design_matrix[key] - design_matrix[key].mean()

    # estimate second level model
    glm_2nd = SecondLevelModel(smoothing_fwhm=8.0)
    glm_2nd = glm_2nd.fit(cmaps, design_matrix=design_matrix)

    for index, covariate in enumerate(covariate_name):
        stats_map = glm_2nd.compute_contrast(covariate, output_type='all')
        t_map = stats_map['stat']
        z_map = stats_map['z_score']

        # write the resulting stat images to file
        tmap_path = pjoin(datasink, '{}_{}_tmap.nii.gz'.format(contrast_id, covariate))
        t_map.to_filename(tmap_path)

        zmap_path = pjoin(datasink,'{}_{}_zmap.nii.gz'.format(contrast_id, covariate))
        z_map.to_filename(zmap_path)


if __name__ == "__main__":
    task = 'game2'
    # subject
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')  # look out
    pid = data['Participant_ID'].to_list()
    sub_list = [p.split('-')[-1] for p in pid]

    print("{} subjects".format(len(sub_list)))

    # configure
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nilearn'
    glm_type = 'cv_hexagon_spct'
    set_id = 'Setall'
    ifold = '6fold'
    templates = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}', 'sub-{}/zmap', '{}_zmap.nii.gz')

    contrast_configs = {'base_spct':['M1', 'M2_corr', 'decision_corr','decision_error', 'correct_error'],
                        'distance_spat': ['M1', 'M2', 'decision', 'M2xdistance', 'decisionxdistance', 'distance'],
                        'distance_spct': ['M1', 'M2_corr', 'decision_corr', 'M2_corrxdistance',
                                          'decision_corrxdistance',
                                          'distance', 'correct_error'],
                        'hexagon_spat': ['M1', 'M2', 'decision', 'm2_hexagon', 'decision_hexagon', 'hexagon'],
                        'hexagon_spct': ['M1', 'M2_corr', 'decision_corr', 'hexagon', 'correct_error','sin','cos'],
                        'cv_hexagon_spct': ['M1', 'M2_corr', 'decision_corr', 'alignPhi'],
                        'hexagon_wpct':['M1','infer_corr','infer_error','hexagon','correct_error'],
                        'hexagon_distance_spat': ['M1', 'M2', 'decision', 'm2_hexagon', 'decision_hexagon', 'hexagon',
                                                  'M2xdistance', 'decisionxdistance', 'distance'],
                        'hexagon_distance_spct': ['M1', 'M2_corr', 'decision_corr', 'decision_error',
                                                  'hexagon','correct_error',
                                                  'M2xdistance', 'decisionxdistance', 'distance'],
                        'cv_train_hexagon_distance_spct':['M1', 'M2_corr','decision_corr',
                                                          'odd_hexagon', 'even_hexagon', 'hexagon', 'correct_error',
                                                          'M2_corrxdistance', 'decision_corrxdistance', 'distance'],
                        'cv_test_hexagon_distance_spct_ECthr3.1':['M1', 'M2_corr', 'M2_error','decision_corr','decision_error',
                                                         'correct_error','alignPhi_even', 'alignPhi_odd','alignPhi',
                                                         'M2_corrxdistance','decision_corrxdistance','distance']
                        }
    contrast_1st = contrast_configs[glm_type]
    data_root = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}', 'group')
    if not os.path.exists(data_root):
        os.mkdir(data_root)

    """
    # run 2nd level GLM
    datasink = os.path.join(data_root, 'mean')
    if not os.path.exists(datasink):
        os.mkdir(datasink)
    Parallel(n_jobs=13)(delayed(run_2nd_ttest)(sub_list, contrast_id, templates, datasink)
                        for contrast_id in contrast_1st)
    """
    # high performance subjects
    hp_data = participants_data.query(f'({task}_fmri>=0.5)and(game1_acc>0.8)and(Age>=18)')  # look out
    hp_sub = [p.split('-')[-1] for p in hp_data['Participant_ID'].to_list()]
    print(len(hp_sub))
    datasink = os.path.join(data_root, 'hp')
    if not os.path.exists(datasink):
        os.mkdir(datasink)
    Parallel(n_jobs=13)(delayed(run_2nd_ttest)(hp_sub, contrast_id, templates, datasink)
                        for contrast_id in contrast_1st)

    """
    hp_data = participants_data.query(f'({task}_fmri>=0.5)and(game1_acc>0.8)')  # look out
    hp_sub = [p.split('-')[-1] for p in hp_data['Participant_ID'].to_list()]
    print(len(hp_sub))
    datasink = os.path.join(data_root, 'hp_all')
    if not os.path.exists(datasink):
        os.mkdir(datasink)
    Parallel(n_jobs=13)(delayed(run_2nd_ttest)(hp_sub, contrast_id, templates, datasink)
                        for contrast_id in contrast_1st)

    # age 8 - 18
    data_juv = data.query("Age<=18")
    juv_pid = data_juv['Participant_ID'].to_list()
    juv_sub_list = [p.split('-')[-1] for p in juv_pid]
    print("subjects number of covariate-age:", len(juv_sub_list))
    age = data_juv['Age'].to_list()
    covariates = {'Age': age}
    datasink = os.path.join(data_root, 'age_to18')
    if not os.path.exists(datasink):
        os.mkdir(datasink)
    Parallel(n_jobs=13)(delayed(run_2nd_covariate)(juv_sub_list, contrast_id, templates, covariates, datasink)
                        for contrast_id in contrast_1st)

    # age 8 - 28
    age = data['Age'].to_list()
    covariates = {'Age': age}
    datasink = os.path.join(data_root, 'age_all')
    if not os.path.exists(datasink):
        os.mkdir(datasink)
    Parallel(n_jobs=13)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
                        for contrast_id in contrast_1st)

    # acc
    if task == 'game1':
        accuracy = data['game1_acc'].to_list()
    elif task == 'game2':
        accuracy = data['game2_test_acc'].to_list()
    else:
        raise Exception("Task name is error!")
    covariates = {'acc': accuracy}
    datasink = os.path.join(data_root, 'acc')
    if not os.path.exists(datasink):
        os.mkdir(datasink)
    Parallel(n_jobs=13)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
                        for contrast_id in contrast_1st)
    
    """
    # model age and acc at same time
    if task == 'game1':
        accuracy = data['game1_acc'].to_list()
    elif task == 'game2':
        accuracy = data['game2_test_acc'].to_list()
    else:
        raise Exception("Task name is error!")
    covariates = {'acc':accuracy}
    age = data['Age'].to_list()
    covariates['Age'] = age
    datasink = os.path.join(data_root,'age_acc')
    if not os.path.exists(datasink): 
        os.mkdir(datasink)
    Parallel(n_jobs=10)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
                        for contrast_id in contrast_1st)
                        
