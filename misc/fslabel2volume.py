# This srcipt covert the annotation of the surface to the volume

import os

# covert the annotation file to label file
mri_annotation2label = 'mri_annotation2label --subject fsaverage --hemi {} --outdir {}'
label_outdir = r'/mnt/data/DCM/tmp/aparc/label'
# call the mri_annotation2label command in terminal
os.system(mri_annotation2label.format('lh',label_outdir))
os.system(mri_annotation2label.format('rh',label_outdir))

# get the reg file(transition matrix file) by tkregister2
temp = '/usr/local/freesurfer/subjects/fsaverage/mri/brain.mgz'
reg_outdir = r'/mnt/data/DCM/tmp/aparc/register.dat'
tkregister2 = 'tkregister2 --mov {} --noedit --s fsaverage --regheader --reg {}'
# call the tkregister2 command in terminal
os.system(tkregister2.format(temp,reg_outdir))

# covert the label file to volume
mri_label2vol = 'mri_label2vol ' \
                '--label {} ' \
                '--temp {} ' \
                '--subject fsaverage ' \
                '--hemi {} ' \
                '--o {} ' \
                '--reg {} ' \
                '--proj frac -1 2.6 .01 --fillthresh 1'
                #'--fill-ribbon '

# get the label file's name from label directory
label_name = os.listdir(label_outdir)

# get affine matrix from the temp to target volume
ori_brain = '/mnt/data/DCM/tmp/aparc/brain.nii.gz'
os.system(f'mri_convert /usr/local/freesurfer/subjects/fsaverage/mri/brain.mgz {ori_brain}')
target_volume = r'/mnt/workdir/DCM/Docs/Mask/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
affine_matix = r'/mnt/data/DCM/tmp/aparc/affine.mat'
flirt = 'flirt -in {} -ref {} -omat {}'
os.system(flirt.format(ori_brain,target_volume,affine_matix))

mri_binarize = 'mri_binarize --dilate 2 --erode 2 --i {} --o {} --min 1'
mask_outdir = r'/mnt/data/DCM/tmp/aparc/mask'
for l in label_name:
    lfile_path = os.path.join(label_outdir,l)
    mask_outpath = os.path.join(mask_outdir,l.replace('.label','')+'.nii.gz')
    if l.startswith('lh'):
        hemi = 'lh'
    elif l.startswith('rh'):
        hemi = 'rh'
    # call the mri_label2vol command in terminal
    print(mri_label2vol.format(lfile_path,temp,hemi,mask_outpath,reg_outdir))
    os.system(mri_label2vol.format(lfile_path,temp,hemi,mask_outpath,reg_outdir))
    #fill the holes inside ROIs is to dilate and erode the binary mask.
    os.system(mri_binarize.format(mask_outpath,mask_outpath))
    # apply the affine matrix to the mask
    os.system('flirt -in {} -ref {} -applyxfm -init {} -out {} -interp nearestneighbour'.format(mask_outpath,target_volume,affine_matix,mask_outpath))
    # The ROI now spills out all over. We use brain mask to tidy this up.
    os.system(f'fslmaths {mask_outpath} -mul /mnt/data/DCM/tmp/aparc/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii {mask_outpath}')

