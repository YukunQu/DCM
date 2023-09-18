import numpy as np
from nilearn.image import load_img,math_img,binarize_img,new_img_like
from scipy.ndimage import binary_dilation,binary_erosion,binary_closing,binary_opening,grey_closing

#%%
# do closing operation for mask
maskpath = r'/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/aparc+aseg2diff.mgz'
mask = load_img(maskpath)
mask = binarize_img(mask,0)
mask_data = mask.get_fdata()
#mask_data = binary_closing(mask_data,iterations=3)
#mask_data = binary_closing(mask_data,iterations=2)
#mask_data = closing(mask_data,iterations=1)

new_mask = new_img_like(mask,mask_data)
# # filter by brain mask
# brain_mask = load_img(r'/mnt/workdir/DCM/Docs/Mask/dmPFC/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz')
# brain_mask = binarize_img(brain_mask)
# img_new = math_img('img1*img2',img1=mask,img2=brain_mask)
savepath = r'/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/aparc+aseg2diff_close1.mgz'
new_mask.to_filename(savepath)


#%%
# get subject list
import os

qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]
sub_list.sort()

for sub in sub_list:
    print(sub)
    maskpath = rf'/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/{sub}/dwi/aparc+aseg2diff.mgz'
    mask = load_img(maskpath)
    mask_data = mask.get_fdata()
    new_mask_data = np.zeros_like(mask_data)

    unique_values = np.unique(mask_data)
    for val in unique_values:
        if val == 0:  # skip background
            continue
        binary_mask = mask_data == val
        closed_mask = binary_closing(binary_mask, iterations=1)
        new_mask_data[closed_mask] = val

    new_mask_data = new_mask_data.astype(np.float32)  # convert to float32
    new_mask = new_img_like(mask, new_mask_data)

    savepath = rf'/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/{sub}/dwi/aparc+aseg2diff_close1.mgz'
    new_mask.to_filename(savepath)