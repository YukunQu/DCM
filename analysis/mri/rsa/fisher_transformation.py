import os
import numpy as np
import pandas as pd
from nilearn.image import load_img,math_img,new_img_like,get_data
from nilearn.masking import apply_mask
from nltools.stats import fisher_r_to_z


def ztransf_img(filepath):
    mask = load_img(r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')
    fmap = load_img(filepath)
    premutation_x = np.random.shuffle(get_data(fmap))
    fmap_mean = apply_mask(fmap, mask).mean()
    fmap_demean = math_img("img-{}".format(fmap_mean),img=fmap)
    fmap_ztransf = math_img("np.arctanh(img)",img=fmap_demean)

    fmap_data = get_data(fmap_ztransf)
    mask_data = get_data(mask)
    fmap_data[mask_data==0] = np.float64('NaN')
    ztransf_map = new_img_like(fmap_ztransf, fmap_data)
    return ztransf_map


if __name__ == "__main__":
    # zscore the 1st level result
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()

    for ifold in range(4,9):
        cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_corr_trials/Setall/6fold/{}/rs-corr_img_coarse_{}fold.nii'
        save_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_corr_trials/Setall/6fold/{}/rs-corr_ztransf_map_coarse_{}fold.nii'
        for sub_id in subjects:
            zscored_map = ztransf_img(cmap_template.format(sub_id,ifold))
            zscored_map.to_filename(save_template.format(sub_id,ifold))
            print("The map of {} have been z-transformed.".format(sub_id))
        print("{}fold have been completed.".format(ifold))