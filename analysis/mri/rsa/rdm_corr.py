import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# calculate similarity between RDMs

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


if __name__ == "__main__":
    import pandas as pd
    fold_r = [[],[],[],[],[]]
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game2_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()

    for sub_id in subjects:
        rdm_6fold = np.load(r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/grid_rsa_corr_trials/Setall/6fold/{}/rsa/{}_grid_RDM_coarse_6fold.npy'.format(sub_id,sub_id))
        rdm2_tri = upper_tri(rdm_6fold)
        for index,i in enumerate(range(4,9)):
            rdm_ifold = np.load(rf'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/grid_rsa_corr_trials/Setall/6fold/{sub_id}/rsa/{sub_id}_grid_RDM_coarse_{i}fold.npy')
            rdm1_tri = upper_tri(rdm_ifold)
            r,p = pearsonr(rdm1_tri,rdm2_tri)
            fold_r[index].append(p)

    import seaborn as sns
    fig,ax = plt.subplots(figsize=(6,5))
    sns.set_style('whitegrid')
    for index,i in enumerate(range(4,9)):
        if i==6:
            continue
        sns.histplot(fold_r[index],label=f'{i}fold')
    plt.legend()

