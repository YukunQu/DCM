import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import pandas as pd
import nibabel as nib
import seaborn as sns
from nilearn import plotting
from nipype.interfaces.base import Bunch
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
import concurrent.futures


def sub_angles(ev_files):
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


def cal_neural_rdm(sub_id):
    # set this path to wherever you saved the folder containing the img-files
    cmap_folder = '/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/grid_rsa_corr_trials/Setall/6fold/{}'
    ev_file_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/grid_rsa_corr_trials/{}/6fold/{}_task-game2_run-{}_events.tsv'
    # get subject's contrast_names
    ev_files = []
    runs = range(1,3)
    for i in runs:  # look out
        ev_files.append(ev_file_tempalte.format(sub_id,sub_id,i))
    sub_angles_set = sub_angles(ev_files)

    con_id_list = list(range(1,len(sub_angles_set)+1))
    # get subject's cmap
    image_paths = [os.path.join(cmap_folder.format(sub_id),
                                'con_{}.nii'.format(str(con_id).zfill(4)))
                   for con_id in con_id_list]

    # load one image to get the dimensions and make the mask
    tmp_img = nib.load(image_paths[0])
    # we infer the mask by looking at non-nan voxels
    mask = ~np.isnan(tmp_img.get_fdata())
    x, y, z = tmp_img.get_fdata().shape

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

    SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='correlation')
    savepath = os.path.join(cmap_folder.format(sub_id),'{}-neural_RDM.hdf5'.format(sub_id))
    SL_RDM.save(savepath,'hdf5',overwrite=True)
    print("The {}'s rdm have been done.".format(sub_id))
    return "The {}'s rdm have been done.".format(sub_id)


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game2_fmri>0.5')  # look out
    subjects = data['Participant_ID'].to_list()

    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        results = [executor.submit(cal_neural_rdm,sub_id) for sub_id in subjects]