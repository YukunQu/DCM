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
    for ifold in ['6fold']:
        task = 'game2'
        # subject
        participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
        participants_data = pd.read_csv(participants_tsv, sep='\t')
        data = participants_data.query(f'{task}_fmri>=0.5')  # look out
        pid = data['Participant_ID'].to_list()
        sub_list = [p.split('-')[-1] for p in pid]

        print("total subjects:{}".format(len(sub_list)))

        # configure
        data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nilearn'
        glm_type = 'distance_spct'
        set_id = 'Setall'
        templates = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}', 'sub-{}/zmap', '{}_zmap.nii.gz')
        #templates = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}', 'sub-{}/rsa',  '{}.nii.gz')

        contrast_configs = {'base_spct':['M1', 'M2_corr', 'decision_corr', 'm2_correct_superiority','decision_correct_superiority'],
                            'hexagon_spct': ['M1', 'M2_corr', 'decision_corr','sin', 'cos','hexagon'],
                            'hexagon_sum':['hexagon'],
                            'hexagon_replace_diff': ['hexagon'],
                            'hexagon_distance_spct': ['M1', 'M2_corr', 'decision_corr', 'decision_error','hexagon','correct_error',
                                                      'M2xdistance', 'decisionxdistance', 'distance'],
                            'cv_train_hexagon_spct':['M1', 'M2_corr','decision_corr','odd_hexagon', 'even_hexagon', 'hexagon'],
                            'cv_test_hexagon_spct': ['M1', 'M2_corr', 'M2_error', 'decision_corr', 'decision_error',
                                                     'alignPhi_odd','alignPhi_even','alignPhi'],
                            'cv_hexagon_spct': ['M1', 'M2_corr', 'decision_corr','alignPhi'],
                            'cv_test_align_spct': ['m2_alignPhi_even','decision_alignPhi_even'],
                            'cv_test_12bin_spct': ['m2_alignPhi','decision_alignPhi','alignPhi'],

                            'distance_spct': ['distance'],
                            'manhd_spct':['m2xmanhd','decisionxmanhd'],
                            '2distance_spct': ['m2xeucd','decisionxeucd','m2xmanhd','decisionxmanhd'],
                            '3distance_spct': ['M1', 'M2_corr', 'decision_corr','eucd','ap','dp'],
                            'ap_distance_spct': ['M1', 'M2_corr', 'decision_corr','ap'],
                            'dp_distance_spct': ['M1', 'M2_corr', 'decision_corr','dp'],
                            'apdp_spct':['ap','dp'],
                            'hexModdistance_spct':['alignxdistance','misalignxdistance','hexModdistance'],

                            'value_spct':['M1', 'M2_corr', 'decision_corr','value','M2_corrxvalue'],
                            'pure_value_spct':['M1', 'M2_corr', 'decision_corr','value'],
                            'ap_spct':['value'],
                            'dp_spct':['value'],
                            'hexModvalue_spct':['alignxvalue','misalignxvalue','value','hexModvalue'],
                            'distance_value_spct':['distance','value1','value2','value_average'],
                            'distance_value_spct_v1':['distance','value1','value2','value','decision_corrxdistance'],
                            '2distance_value_spct':['M1', 'M2_corr', 'decision_corr','value','eucd','manhd'],

                            'grid_rsa':['rsa_img_coarse_6fold','rsa_zscore_img_coarse_6fold'],
                            'grid_rsa_corr_trials':['rsa_img_coarse_6fold','rsa_zscore_img_coarse_6fold'],
                            'map_rsa':['cmap_rsa_img','cmap_rsa_zscore_img'],
                            'map_rsa_spat':['cmap_rsa_img','cmap_rsa_ztransf_img'],
                            'base_diff':['M1', 'M2_corr', 'decision_corr', 'correct_error'],
                            'hexagon_diff':['M1', 'M2_corr', 'decision_corr', 'correct_error','hexagon'],
                            'distance_diff':['distance'],
                            'value_diff':['M1', 'M2_corr', 'decision_corr','correct_error','value'],
                            '2distance_diff':['M1', 'M2_corr', 'decision_corr','m2xeucd', 'decisionxeucd', 'm2xmanhd', 'decisionxmanhd', 'correct_error'],
                            'distance_value_diff':['distance_value_diff'],

                            'hexagon_center_spct':['M1', 'M2_corr', 'decision_corr', 'correct_error', 'hexagon','sin','cos'],
                            'distance_center_spct':['M1', 'M2_corr', 'decision_corr', 'correct_error', 'M2_corrxdistance'],
                            'cv_hexagon_center_spct':['M1', 'M2_corr', 'decision_corr','alignPhi']
                            }
        print('glm',glm_type,'start.')
        contrast_1st = contrast_configs[glm_type]

        # create output directory
        data_root = pjoin(data_root, f'{task}/{glm_type}/{set_id}/{ifold}', 'group_zmap')
        os.makedirs(data_root,exist_ok=True)

        # # mean effect of all subjects
        datasink = os.path.join(data_root, 'mean')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_ttest)(sub_list, contrast_id, templates, datasink)
                            for contrast_id in contrast_1st)

        # mean effect of high performance subjects
        hp_data = participants_data.query(f'({task}_fmri>=0.5)and(game2_test_acc>0.80)')  # look out
        hp_sub = [p.split('-')[-1] for p in hp_data['Participant_ID'].to_list()]
        print("Game2's high performance subjects:",len(hp_sub))
        datasink = os.path.join(data_root, 'game2_hp_all')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_ttest)(hp_sub, contrast_id, templates, datasink)
                            for contrast_id in contrast_1st)

        hp_data = participants_data.query(f'({task}_fmri>=0.5)and(game1_acc>0.80)')  # look out
        hp_sub = [p.split('-')[-1] for p in hp_data['Participant_ID'].to_list()]
        print("Both games' high performance subjects:",len(hp_sub))
        datasink = os.path.join(data_root, 'game1_hp_all')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_ttest)(hp_sub, contrast_id, templates, datasink)
                            for contrast_id in contrast_1st)

        # mean effect of adult subjects
        adult_data = participants_data.query(f'({task}_fmri>=0.5)and(Age>=18)')  # look out
        adult_sub = [p.split('-')[-1] for p in adult_data['Participant_ID'].to_list()]
        print("adult subjects:",len(adult_sub))
        datasink = os.path.join(data_root, 'adult')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_ttest)(adult_sub, contrast_id, templates, datasink)
                            for contrast_id in contrast_1st)

        # The accuracy effect for all subjects(8-25)
        if task == 'game1':
            accuracy = data['game1_acc'].to_list()
        elif task == 'game2':
            accuracy = data['game2_test_acc'].to_list()
        else:
            raise Exception("Task name is error!")
        covariates = {'acc': accuracy}
        datasink = os.path.join(data_root, 'acc')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
                            for contrast_id in contrast_1st)

        # The age effect for all subjects(8-25)
        age = data['Age'].to_list()
        covariates = {'Age': age}
        print("The subjects number of covariate-age(8-25):", len(sub_list))
        datasink = os.path.join(data_root, 'age_all')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
                            for contrast_id in contrast_1st)

        # The age and accuracy effect for subjects(8-25)
        covariates = {'age': age, 'acc': accuracy}
        print("subjects number of age-acc effect:", len(sub_list))
        datasink = os.path.join(data_root, 'age_acc')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
                            for contrast_id in contrast_1st)

        # # The age effect for subjects(8-17)
        data_juv = data.query("Age<18")
        juv_pid = data_juv['Participant_ID'].to_list()
        juv_sub_list = [p.split('-')[-1] for p in juv_pid]
        print("subjects number of covariate-age(8-17):", len(juv_sub_list))
        age = data_juv['Age'].to_list()
        covariates = {'Age': age}
        datasink = os.path.join(data_root, 'age_to17')
        os.makedirs(datasink,exist_ok=True)
        Parallel(n_jobs=25)(delayed(run_2nd_covariate)(juv_sub_list, contrast_id, templates, covariates, datasink)
                            for contrast_id in contrast_1st)

        # The covariate effect between the behavioral difference and neural difference
        # datasink = os.path.join(data_root, 'behav_diff')
        # acc_diff = data['game2_test_acc'] - data['game1_acc'].to_list()
        # covariates = {'beh_diff': acc_diff}
        # os.makedirs(datasink,exist_ok=True)
        # Parallel(n_jobs=25)(delayed(run_2nd_covariate)(sub_list, contrast_id, templates, covariates, datasink)
        #                     for contrast_id in contrast_1st)