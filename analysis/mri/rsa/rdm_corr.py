import numpy as np
from scipy.stats import pearsonr

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


rdm1 = np.load(r'/mnt/data/DCM/result_backup/2023.1.2/game1/grid_rsa_8mm/Setall/6fold/sub-200/sub-200_grid_RDM_coarse_6fold.npy')
rdm2 = np.load(r'/mnt/data/DCM/result_backup/2023.1.2/game1/grid_rsa_8mm/Setall/6fold/sub-200/sub-200_grid_RDM_coarse_8fold.npy')

rdm1_tri = upper_tri(rdm1)
rdm2_tri = upper_tri(rdm2)

r,p = pearsonr(rdm1_tri,rdm2_tri)