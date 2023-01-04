import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import pandas as pd
import nibabel as nib
import seaborn as sns
from nilearn import plotting

from rsatoolbox.rdm import RDMs
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from analysis.mri.rsa.grid_model_rdm import sub_angles
import concurrent.futures


def cal_neural_rdm(sub_id):
    # set this path to wherever you saved the folder containing the img-files
    cmap_folder = '/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/grid_rsa_8mm/Setall/6fold/{}'
    ev_file_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/grid_rsa/{}/6fold/{}_task-game2_run-{}_events.tsv'

    # get subject's contrast_names
    ev_files = []
    for i in range(1,3):
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        results = [executor.submit(cal_neural_rdm,sub_id) for sub_id in subjects]