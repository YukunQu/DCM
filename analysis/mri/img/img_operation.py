from nilearn.image import load_img,math_img,binarize_img,new_img_like
from scipy.ndimage import binary_dilation,binary_erosion,binary_closing,binary_opening


# do closing operation for mask
maskpath = r'/mnt/workdir/DCM/Docs/Mask/PCC/PCCk3_MNI152Nl.nii.gz'
mask = load_img(maskpath)
mask = binarize_img(mask,0)
mask_data = mask.get_fdata()
#mask_data = binary_closing(mask_data,iterations=3)
#mask_data = binary_closing(mask_data,iterations=2)
mask_data = binary_opening(mask_data,iterations=4)
mask = new_img_like(mask,mask_data)
# filter by brain mask
brain_mask = load_img(r'/mnt/workdir/DCM/Docs/Mask/dmPFC/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz')
brain_mask = binarize_img(brain_mask)
img_new = math_img('img1*img2',img1=mask,img2=brain_mask)
savepath = r'/mnt/workdir/DCM/Docs/Mask/PCC/PCCk3_MNI152Nl_o4.nii.gz'
img_new.to_filename(savepath)