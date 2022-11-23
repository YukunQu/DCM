import os
import shutil
import numpy as np
from analysis.mri.img.mask_img import mask_img


subject_list = [36,46,62,76,78,79,121,145,148]
subject_list = [str(s).zfill(3) for s in subject_list]

anat_list = [r'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap/' \
             rf'sub-{s}/anat/sub-{s}_desc-preproc_T1w.nii.gz' for s in subject_list]

anat_mask_list = [r'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap/' \
                  rf'sub-{s}/anat/sub-{s}_desc-brain_mask.nii.gz' for s in subject_list]

anat_out_list = [r'/mnt/workdir/DCM/BIDS/derivatives/fsl/preprocessed/smooth_6/' \
                 rf'sub-{s}/anat/sub-{s}_T1w.nii.gz' for s in subject_list]

brain_out_list = [r'/mnt/workdir/DCM/BIDS/derivatives/fsl/preprocessed/smooth_6/' \
                 rf'sub-{s}/anat/sub-{s}_T1w_brain.nii.gz' for s in subject_list]

# move T1w to fsl directory
for a,a_out in zip(anat_list,anat_out_list):
    shutil.copy(a,a_out)
    print(a_out,'finished.')

# stripping skull
for a,a_mask,b_out in zip(anat_list,anat_mask_list,brain_out_list):
    mask_img(a,a_mask,b_out)
    print(b_out,'finished.')