import os
import numpy as np
import pandas as pd
from os.path import join
from nilearn.image import load_img, concat_imgs
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix


def load_ev(event_path):
    event = pd.read_csv(event_path, sep='\t')
    event_condition = event[['onset', 'duration', 'trial_type', 'modulation']]
    return event_condition


def prepare_data(subj,ifold,configs,load_ev,concat_runs=True,despiking=True):
    """
    prepare functional images and design matrixs
    :param subj: sub_id
    :param ifold: event mid-directory
    :param configs: A series of parameters:TR,func_dir,event_dir,task,glm_type,run_list,
                                           func_name,events_name,regressor_file
    :param load_ev: load event file and generate parametric modulation
    :param concat_runs: concatenate all runs into a big one.
    :param despiking: volume censoring to remove volumes with high motion
    :return:funcitonal list;design matirx
    """
    tr = configs['TR']
    func_dir = configs['func_dir']
    event_dir = configs['event_dir']

    task = configs['task']
    glm_type = configs['glm_type']
    run_list = configs['run_list']

    func_name = configs['func_name']
    events_name = configs['events_name']
    regressor_file = configs['regressor_name']
    ifold = '{}fold'.format(ifold)

    functional_imgs = []
    design_matrices = []
    for i, run_id in enumerate(run_list):
        # load image
        func_path = join(func_dir, f'sub-{subj}',  func_name.format(subj, run_id))
        func_img = load_img(func_path)
        functional_imgs.append(func_img)

        # load event
        event_path = join(event_dir, task, glm_type, f'sub-{subj}', ifold, events_name.format(subj, run_id))
        event = load_ev(event_path)

        # load motion
        add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                         'csf', 'white_matter']
        confound_file = os.path.join(func_dir, f'sub-{subj}', 'func', regressor_file.format(subj, run_id))
        confound_factors = pd.read_csv(confound_file, sep="\t")
        motion = confound_factors[add_reg_names]
        motion = motion.fillna(0.0)

        # remove bad time
        if despiking:
            fd = confound_factors['framewise_displacement']
            outlier_time = fd[fd > 0.5].index
            outlier_reg = {}
            for index,ot in enumerate(outlier_time):
                otreg = np.zeros(fd.shape[0])
                otreg[ot] = 1
                outlier_reg['outlier{}'.format(index+1)] = otreg
            outlier_reg = pd.DataFrame(outlier_reg)
            motion = pd.concat([motion,outlier_reg],axis=1)

        # creat design matrix
        n_scans = func_img.shape[-1]
        frame_times = np.arange(n_scans) * tr
        high_pass_fre = 1 / 100
        design_matrix = make_first_level_design_matrix(
            frame_times,
            event,
            hrf_model='spm',
            drift_model=None,
            high_pass=high_pass_fre, add_regs=motion, add_reg_names=add_reg_names)
        design_matrices.append(design_matrix)

    # concatenate runs
    if concat_runs:
        functional_imgs = concat_imgs(functional_imgs)

        new_design_matrices = []
        for run_label, dm in enumerate(design_matrices):
            dm_columns = dm.columns
            new_dm_columns = ['s' + str(run_label + 1) + '_' + c for c in dm_columns]
            dm.columns = new_dm_columns
            new_design_matrices.append(dm)
        design_matrices = pd.concat(new_design_matrices, axis=0, ignore_index=True)
        design_matrices = design_matrices.fillna(0.0)
    #design_matrices.to_csv(r"/mnt/workdir/DCM/tmp/orth2.csv")
    return functional_imgs, design_matrices


def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def get_reg_index(design_matrix, target_name):
    target_index = []
    for i, reg_name in enumerate(design_matrix.columns):
        if target_name in reg_name:
            target_index.append(i)
    if len(target_index) == 0:
        print("The {} don't have regressor.".format(target_name))
    return target_index


def first_level_glm(datasink, run_imgs, design_matrices, set_contrasts):
    """
    run first level GLM and save result to datasink directory.
    :param datasink: output directory
    :param run_imgs: a functional image or a list contain multi-runs of functional iamges
    :param design_matrices:a design matrix or a list contain multi-runs of design_matrix
    :param set_contrasts: set_contrast function to generate contrast from design matrix
    """
    if not os.path.exists(datasink):
        os.makedirs(datasink)
        for dir in ['cmap', 'stat_map', 'zmap']:
            if not os.path.exists(os.path.join(datasink, dir)):
                os.mkdir(os.path.join(datasink, dir))

    # fit first level glm to estimate mean orientation
    mni_mask = r'/mnt/workdir/DCM/Docs/Mask/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    fmri_glm = FirstLevelModel(t_r=3.0, slice_time_ref=0.5, hrf_model='spm',
                               drift_model=None, high_pass=1/ 100, mask_img=mni_mask,
                               smoothing_fwhm=8, verbose=1, n_jobs=1)
    fmri_glm = fmri_glm.fit(run_imgs, design_matrices=design_matrices)

    # define contrast
    contrasts = set_contrasts(design_matrices)

    # statistics inference
    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the two sessions

        stats_map = fmri_glm.compute_contrast(contrast_val, output_type='all')
        c_map = stats_map['effect_size']
        stat_map = stats_map['stat']
        z_map = stats_map['z_score']
        # write the resulting stat images to file
        c_image_path = join(datasink, 'cmap', '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)

        s_image_path = join(datasink, 'stat_map', '%s_smap.nii.gz' % contrast_id)
        stat_map.to_filename(s_image_path)

        z_image_path = join(datasink, 'zmap', '%s_zmap.nii.gz' % contrast_id)
        z_map.to_filename(z_image_path)