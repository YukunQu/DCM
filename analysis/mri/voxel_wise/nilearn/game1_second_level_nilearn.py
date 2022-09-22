import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm import threshold_stats_img


def second_level_covariate(subjects,task,glm_type,contrast_id,covariates,datasink,centring=False):
    # Select cmaps
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nilearn'
    cmap_template = pjoin(data_root, f'{task}/{glm_type}/Setall/6fold','sub-{}', f'{contrast_id}.nii.gz')
    cmap_files = []
    for subj_id in subjects:
        cmap_files.append(cmap_template.format(subj_id))

    # Set design matrix
    covariate_name = covariates.keys()
    extra_info_subjects = pd.DataFrame(covariates)
    extra_info_subjects['subject_label'] = subjects

    design_matrix = make_second_level_design_matrix(subjects,extra_info_subjects)
    for key in covariate_name:
        design_matrix[key] = design_matrix[key] - design_matrix[key].mean()

    # estimate second level model
    mni_mask = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    model = SecondLevelModel(smoothing_fwhm=5.0,mask_img=mni_mask)
    model.fit(cmap_files, design_matrix=design_matrix)

    if not os.path.exists(datasink):
        os.mkdir(datasink)

    for index, covariate in enumerate(covariate_name):
        stats_map = model.compute_contrast(covariate, output_type='all')
        t_map = stats_map['stat']
        z_map = stats_map['z_score']

        # write the resulting stat images to file
        tmap_path = pjoin(datasink,'{}_{}_tmap.nii.gz'.format(contrast_id,covariate))
        t_map.to_filename(tmap_path)

        zmap_path = pjoin(datasink,'{}_{}_zmap.nii.gz'.format(contrast_id,covariate))
        z_map.to_filename(zmap_path)


def second_covariate():
    task = 'game1'  # look out
    glm_type = 'separate_hexagon'

    # load behavioural results
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')  # look out
    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('-')[-1] for p in pid]
    accuracy = data['game1_acc'].to_list()  # look out
    #age = data['Age'].to_list()
    #covariate = {'Age':age}
    covariate = {'Acc':accuracy}
    datasink = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/separate_hexagon/Setall/group'

    # load contrast data
    contrast_1st = ['m2_hexagon_tmap','decision_hexagon_tmap','hexagon_tmap']
    for contrast_id in contrast_1st:
        second_level_covariate(subject_list,task,glm_type,contrast_id,covariate,datasink,centring=True)


if __name__ == "__main__":
    """
    task = 'game1'
    glm_type = 'separate_hexagon'

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query('game1_fmri==1')  # look out

    adult_data = data.query('Age>18')
    adolescent_data = data.query('12<Age<=18')
    children_data = data.query('Age<=12')
    hp_data = data.query('game1_acc>=0.8')

    print("Participants:", len(data))
    print("Adult:",len(adult_data))
    print("Adolescent:",len(adolescent_data))
    print("Children:", len(children_data))
    print("High performance:",len(hp_data),"({} adult)".format(len(hp_data.query('Age>18'))))

    pid = data['Participant_ID'].to_list()
    subject_list = [p.split('_')[-1] for p in pid]
    #subject_list.remove('079')
    print(len(subject_list))

    # Select cmaps
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nilearn'
    cmap_template = pjoin(data_root, f'{task}/{glm_type}/Setall/6fold','sub-{}', 'hexagon_zmap.nii.gz')
    cmap_files = []
    for subj_id in subject_list:
        cmap_files.append(cmap_template.format(subj_id))

    design_matrix = pd.DataFrame([1] * len(cmap_files), columns=['intercept'])

    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(cmap_files,design_matrix=design_matrix)

    stats_map = second_level_model.compute_contrast(output_type='all')

    t_map = stats_map['stat']
    z_map = stats_map['z_score']
    # write the resulting stat images to file
    datasink = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/{}/{}/Setall/group'.format(task,glm_type)
    tmap_path = pjoin(datasink,'hexagon_tmap.nii.gz')
    t_map.to_filename(tmap_path)

    zmap_path = pjoin(datasink,'hexagon_zmap.nii.gz')
    z_map.to_filename(zmap_path)
    """
    second_covariate()