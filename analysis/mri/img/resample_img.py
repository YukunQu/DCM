from nilearn.image import resample_img,resample_to_img


def resample_to_mni152nl(ori_img,savepath):
    template = '/mnt/workdir/DCM/Docs/Mask/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
    resampled_img = resample_to_img(ori_img,template,interpolation='nearest')
    resampled_img.to_filename(savepath)


ori_img = '/mnt/workdir/DCM/Docs/Mask/dmPFC/BiMPFC_NeuroSynth.nii.gz'
savepath = '/mnt/workdir/DCM/Docs/Mask/dmPFC/BiMPFC_NeuroSynth_MNINL.nii.gz'
resample_to_mni152nl(ori_img,savepath)


#