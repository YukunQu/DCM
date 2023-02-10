import os
import shutil
import glob
import numpy as np
import pandas as pd
from subprocess import call
from nipype.interfaces.fsl import maths
from nilearn.image import mean_img,resample_to_img
from analysis.mri.img.mask_img import mask_img


def fmriprep2fsl(sub_dir,out_dir,smooth=0):
    sub_id = sub_dir.split('/')[-1]
    func_template = os.path.join(sub_dir,'func','sub-*_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz')
    func_list = glob.glob(func_template)
    func_list.sort()
    for func_file in func_list:
        func_name = func_file.split('/')[-1]
        print(func_name,'start!')
        func_name = func_name.replace('.nii.gz','.ica')
        out_func_dir = os.path.join(out_dir,func_name)
        if not os.path.exists(out_func_dir):
            os.mkdir(out_func_dir)
        out_func_path = os.path.join(out_func_dir,'filtered_func_data.nii.gz')
        shutil.copy(func_file,out_func_path)

        # covert example_func.nii.gz
        example_func_file = func_file.replace('_desc-preproc_bold.nii.gz','_boldref.nii.gz')
        reg_dir = os.path.join(out_func_dir,'reg')
        if not os.path.exists(reg_dir):
            os.mkdir(reg_dir)
        out_example_func_path = os.path.join(reg_dir,'example_func.nii.gz')
        if smooth==0:
            shutil.copy(example_func_file,out_example_func_path)
        else:
            pass

        # generate mean_func.nii.gz
        mean_func = mean_img(func_file)
        mean_func_outpath = os.path.join(out_func_dir,'mean_func.nii.gz')
        mean_func.to_filename(mean_func_outpath)

        # covert T1w images and T1w mask
        mask_file = glob.glob(os.path.join(sub_dir,'anat',f'{sub_id}_desc-brain_mask.nii.gz'))[0]
        mask_outpath = os.path.join(out_func_dir,'mask.nii.gz')
        resampled_mask = resample_to_img(mask_file,func_file,'nearest')  # resample mask to func image
        resampled_mask.to_filename(mask_outpath)
        di = maths.DilateImage(in_file=mask_outpath,
                               operation="mean",
                               kernel_shape="sphere",
                               kernel_size=3,
                               out_file=mask_outpath)
        di.run()

        anat_file = glob.glob(os.path.join(sub_dir,'anat',f'{sub_id}_desc-preproc_T1w.nii.gz'))[0]
        anat_outpath = os.path.join(out_func_dir,'reg','highres.nii.gz')
        mask_img(anat_file,mask_file,anat_outpath)  # skull stripping

        # motion parameters
        confound_file = func_file.replace('_space-T1w_desc-preproc_bold.nii.gz','_desc-confounds_timeseries.tsv')
        motions_df = pd.read_csv(confound_file,sep='\t')
        motion_columns = ['rot_x','rot_y','rot_z','trans_x','trans_y','trans_z']
        motions = motions_df[motion_columns].to_numpy()
        mc_dir = os.path.join(out_func_dir,'mc')
        if not os.path.exists(mc_dir):
            os.mkdir(mc_dir)
        motion_outpath = os.path.join(mc_dir,'prefiltered_func_data_mcf.par')
        np.savetxt(motion_outpath,motions,delimiter='  ')

        # create affine matrix
        mat_outpath = os.path.join(out_func_dir,'reg','highres2example_func.mat')
        flirt_cmd = 'flirt -in {} -ref {} -omat {}'.format(anat_outpath,out_example_func_path,mat_outpath)
        print("Command:",flirt_cmd)
        call(flirt_cmd,shell=True)


if __name__ =="__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')  #look out
    data = data.query("(game1_acc>=0.8)and(Age>=18)")
    subject_list = data['Participant_ID'].to_list()
    for sub in subject_list:
        sub_dir = rf'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap/{sub}'
        out_dir = os.path.join(sub_dir,'fsl_smooth0')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        fmriprep2fsl(sub_dir,out_dir)