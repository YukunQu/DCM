import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.colors
import matplotlib.pyplot as plt
from rsatoolbox.rdm.rdms import load_rdm
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.util.searchlight import  evaluate_models_searchlight
from nilearn.image import new_img_like
import concurrent.futures


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

def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap


def calc_rs_map(sub_id):
    # set path
    neural_RDM_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_8mm/Setall/6fold/' \
                      r'{}/{}-neural_RDM.hdf5'.format(sub_id,sub_id)
    gird_RDM_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_8mm/Setall/6fold/' \
                    r'{}/{}_grid_RDM_coarse.npy'.format(sub_id,sub_id)
    rsmap_savepath = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_8mm/Setall/6fold/' \
                     r'{}/rs-corr_img_coarse.nii'.format(sub_id,sub_id)

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
    tmp_img = nib.load(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_8mm/Setall/6fold/'
                       r'{}/con_0001.nii'.format(sub_id))
    # we infer the mask by looking at non-nan voxels
    mask = ~np.isnan(tmp_img.get_fdata())
    x, y, z = mask.shape
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(neural_RDM.rdm_descriptors['voxel_index'])] = eval_score
    RDM_brain = RDM_brain.reshape([x, y, z])

    corr_img = new_img_like(tmp_img, RDM_brain)
    corr_img.to_filename(rsmap_savepath)
    print("The calculation of {} have been done.".format(sub_id))

def list_to_chunk(orignal_list,chunk_volume=30):
    chunk_list = []
    chunk = []
    for i, element in enumerate(orignal_list):
        chunk.append(element)
        if len(chunk) == chunk_volume:
            chunk_list.append(chunk)
            chunk = []
        elif i == (len(orignal_list) - 1):
            chunk_list.append(chunk)
        else:
            continue
    return chunk_list


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()
    subjects_chunk = list_to_chunk(subjects)

    for sub_chunk in subjects_chunk:
        with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
            results = [executor.submit(calc_rs_map,sub_id) for sub_id in sub_chunk]