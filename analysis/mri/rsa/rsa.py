import os
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.colors
import matplotlib.pyplot as plt
from rsatoolbox.rdm.rdms import load_rdm
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.util.searchlight import evaluate_models_searchlight

from nilearn.image import new_img_like
import concurrent.futures
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk


def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


def calc_rs_map(sub_id, ifold):
    """
    calculate correaltion between neural RDM and model RDM
    :param sub_id:
    :param ifold:
    :return:
    """
    # set path
    default_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/grid_rsa_corr_trials/Setall/6fold'
    neural_RDM_path = os.path.join(default_dir,'{}/rsa/{}-neural_RDM_cmap.hdf5'.format(sub_id,sub_id))
    gird_RDM_path = os.path.join(default_dir,'{}/rsa/{}_grid_RDM_coarse_{}fold.npy'.format(sub_id,sub_id,ifold))
    rsmap_savepath = os.path.join(default_dir,'{}/rsa/rsa_cmap_img_coarse_{}fold.nii.gz'.format(sub_id,ifold))

    #  load neural RDM for each voxel
    neural_RDM = load_rdm(neural_RDM_path)
    # load grid model
    grid_RDM = np.load(gird_RDM_path)
    grid_model = ModelFixed('Grid RDM', upper_tri(grid_RDM))
    # evaluate
    eval_results = evaluate_models_searchlight(neural_RDM, grid_model, eval_fixed, method='corr', n_jobs=3)

    # get the evaulation score for each voxel
    # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
    eval_score = [np.float64(e.evaluations) for e in eval_results]

    # Create an 3D array, with the size of mask, and
    mni_mask = r'/mnt/workdir/DCM/docs/Mask/res-02_desc-brain_mask.nii'
    mask_img = nib.load(mni_mask)
    mask = mask_img.get_fdata()
    x, y, z = mask.shape

    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(neural_RDM.rdm_descriptors['voxel_index'])] = eval_score
    RDM_brain = RDM_brain.reshape([x, y, z])

    corr_img = new_img_like(mask_img, RDM_brain)
    corr_img.to_filename(rsmap_savepath)
    print("The calculation of {}-{} have been done.".format(ifold,sub_id))


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()
    subjects_chunk = list_to_chunk(subjects,70)

    for ifold in [6]:
        print('-------------{} fold start.-------------------' .format(ifold))
        for sub_chunk in subjects_chunk:
            with concurrent.futures.ProcessPoolExecutor(max_workers=70) as executor:
                results = [executor.submit(calc_rs_map,sub_id,ifold) for sub_id in sub_chunk]
