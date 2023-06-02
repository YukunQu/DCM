import os.path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import binarize_img
from nipype.interfaces.base import Bunch
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from joblib import Parallel, delayed


def get_sub_angles_spm(ev_files):
    runs_info = []
    for ev_file in ev_files:
        onsets = []
        conditions = []
        durations = []

        ev_info = pd.read_csv(ev_file, sep='\t')
        for group in ev_info.groupby('trial_type'):
            condition = group[0]
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        run_info = Bunch(conditions=conditions, onsets=onsets, durations=durations)
        runs_info.append(run_info)

    conditions = []
    for run_info in runs_info:
        conditions.extend(run_info.conditions)
    conditions_names = list(set(conditions))
    conditions_names.sort()
    for non_angle_reg in ['M1','M2_error','decision']:
        if non_angle_reg in conditions_names:
            conditions_names.remove(non_angle_reg)
        else:
            print(ev_files,':',non_angle_reg)
    sub_angles_set = [float(c) for c in conditions_names]
    return sub_angles_set


def get_sub_angles_nilearn(ev_files):
    """
    get angle regressors name from event files
    :param ev_files: a list of event files path
    :return:
    """
    regressors_name = []
    for ev_file in ev_files:
        ev_info = pd.read_csv(ev_file, sep='\t')
        regressors_name.extend(ev_info['trial_type'].to_list())
    angle_con_names = list(set(regressors_name))

    # remove other regressors.
    for non_angle_reg in ['M1','M2_error','decision']:
        if non_angle_reg in angle_con_names:
            angle_con_names.remove(non_angle_reg)
        else:
            print(ev_files,"don't have",non_angle_reg)

    # sort angle as value
    angle_names = [int(acn.replace('angle','')) for acn in angle_con_names]
    angle_names.sort()
    angle_con_names = ['angle'+str(a) for a in angle_names]
    return angle_con_names


def get_sub_pos(ev_files):
    """
    get angle regressors name from event files
    :param ev_files: a list of event files path
    :return:
    """
    regressors_name = []
    for ev_file in ev_files:
        ev_info = pd.read_csv(ev_file, sep='\t')
        regressors_name.extend(ev_info['trial_type'].to_list())
    angle_con_names = list(set(regressors_name))

    # remove other regressors.
    for non_angle_reg in ['error','decision_corr','decision_error']:
        if non_angle_reg in angle_con_names:
            angle_con_names.remove(non_angle_reg)
        else:
            print(ev_files,"don't have",non_angle_reg)
    # sort angle as value
    angle_con_names.sort()
    return angle_con_names


def cal_neural_rdm(sub_id):
    """
    using searchilight calculate neural RDM for different angles
    :param sub_id:
    :return:
    """
    # get subject's contrast_names(angles)
    ev_files = []
    ev_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/' \
                  r'game1/map_rsa/{}/6fold/{}_task-game1_run-{}_events.tsv'  # look out
    runs = range(1,7)  # look out
    for i in runs:
        ev_files.append(ev_tempalte.format(sub_id,sub_id,i))
    #con_names = get_sub_angles_nilearn(ev_files)
    con_names = get_sub_pos(ev_files)

    # get subject's cmap
    cmap_folder = '/mnt/workdir/DCM/BIDS/derivatives/Nilearn/' \
                  'game1/map_rsa/Setall/6fold/{}'
    image_paths = [os.path.join(cmap_folder.format(sub_id),'cmap/{}_cmap.nii.gz'.format(con_id))
                   for con_id in con_names]

    # load one image to get the dimensions and make the mask
    mni_mask = r'/mnt/workdir/DCM/Docs/Mask/res-02_desc-brain_mask.nii'
    mask_img = nib.load(mni_mask)
    mask = mask_img.get_fdata()
    x, y, z = mask.shape

    # loop over all images
    data = np.zeros((len(image_paths), x, y, z))
    for x, im in enumerate(image_paths):
        data[x] = nib.load(im).get_fdata()

    # only one pattern per image
    image_value = np.arange(len(image_paths))

    # get searchlight
    centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=0.5)

    data_2d = data.reshape([data.shape[0], -1])
    data_2d = np.nan_to_num(data_2d)

    SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='mahalanobis')
    savepath = os.path.join(cmap_folder.format(sub_id),'rsa')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    savepath = os.path.join(savepath,'{}-neural_cmap_RDM.hdf5'.format(sub_id))
    SL_RDM.save(savepath,'hdf5',overwrite=True)
    print("The {}'s rdm have been done.".format(sub_id))
    return "The {}'s rdm have been done.".format(sub_id)


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()

    subjects_chunk = list_to_chunk(subjects,50)
    for chunk in subjects_chunk:
        results_list = Parallel(n_jobs=50,backend="multiprocessing")(delayed(cal_neural_rdm)(subj) for subj in chunk)