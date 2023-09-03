from nilearn.image import resample_to_img

# Resample the image to the same resolution as the target
img = r'/mnt/workdir/DCM/Docs/Mask/RSC/Tracking the Emergence of Location-based Spatial Representations in Human Scene-Selective Cortex/0_GroupNormedRoi-rRSC.nii.gz'
target_img = r'/mnt/workdir/DCM/Docs/Mask/RSC/Tracking the Emergence of Location-based Spatial Representations in Human Scene-Selective Cortex/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
resampled_img = resample_to_img(img, target_img,interpolation='nearest')
resampled_img.to_filename(r'/mnt/workdir/DCM/Docs/Mask/RSC/Tracking the Emergence of Location-based Spatial Representations in Human Scene-Selective Cortex/MNI152Nl-rRSC.nii.gz')