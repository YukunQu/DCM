import os
import subprocess

mnil_template = '/usr/local/fsl/etc/flirtsch/ident.mat'
mninl_template = '/usr/local/fsl/etc/flirtsch/ident.mat'

def reg_MNIL2MNINL(ori_img):
    mnil= r'/mnt/workdir/DCM/Docs/Mask/dmPFC/MNI152_T1_2mm_brain.nii.gz'
    mninl = r'/mnt/workdir/DCM/Docs/Mask/dmPFC/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
    mnil_new = os.path.join(os.path.dirname(ori_img), 'MNINonL_' + os.path.basename(mnil))
    affine_mat = os.path.join(os.path.dirname(ori_img), 'affine.mat')
    cmd1 = f'flirt -in {mnil} -ref {mninl} -out {mnil_new} -omat {affine_mat} -interp nearestneighbour'
    subprocess.call(cmd1, shell=True)
    out_img = os.path.join(os.path.dirname(ori_img), 'MNINonL_' + os.path.basename(ori_img))
    cmd2 = f'flirt -in {ori_img} -ref {mninl} -applyxfm -init {affine_mat} -out {out_img} -interp nearestneighbour'
    subprocess.call(cmd2, shell=True)


if __name__ == '__main__':
    ori_img = r'/mnt/workdir/DCM/Docs/Mask/dmPFC/BN_Atlas_246_2mm.nii.gz'
    reg_MNIL2MNINL(ori_img)