import os
import glob
import nibabel as nib



def check_fmri_completeness(fmrif, f,vthr=140):
    # Load the fMRI data header
    fmri_header = nib.load(fmrif).header

    # Get the number of volumes from the header
    num_volumes = fmri_header.get_data_shape()[3]

    # Check if the number of volumes is greater than 140
    if num_volumes > vthr:
        f.write(f"{fmrif} is fine.\n")
    else:
        f.write(f"The fMRI data file {fmrif} has less than or equal to {vthr} volumes.\n")


def check_t1_completeness(t1f,f,shape=(192,448,512)):
    # Load the fMRI data header
    t1_header = nib.load(t1f).header

    # Get the number of volumes from the header
    t1_shape = t1_header.get_data_shape()

    # Check if the number of volumes is greater than 140
    if t1_shape == shape:
        f.write(f"{t1f} is fine.\n")
    else:
        f.write(f"The shape of {t1f} not equal to {shape}.\n")


if __name__ == "__main__":
    # Set data template
    game_template = r'/mnt/workdir/DCM/BIDS/*/func/*_task-game*_run-*_bold.nii.gz'

    # Get a list of fMRI data files in the directory
    fmri_files = glob.glob(os.path.join(game_template))
    fmri_files.sort()

    thr = 100

    # Open the output file
    with open('/mnt/workdir/DCM/tmp/check_fmri_data_completeness_rest_output.txt', 'w') as f:
        # Loop through the fMRI data files
        for fmri_file in fmri_files:
            check_fmri_completeness(fmri_file,f,thr)

    # Set data template
    rest_template = r'/mnt/workdir/DCM/BIDS/*/func/*_task-rest_run-*_bold.nii.gz'

    # Get a list of fMRI data files in the directory
    fmri_files = glob.glob(os.path.join(rest_template))
    fmri_files.sort()

    thr = 90

    # Open the output file
    with open('/mnt/workdir/DCM/tmp/check_fmri_data_completeness_rest_output.txt', 'w') as f:
        # Loop through the fMRI data files
        for fmri_file in fmri_files:
            check_fmri_completeness(fmri_file,f,thr)

    # Set data template
    t1_template = r'/mnt/workdir/DCM/BIDS/*/anat/*_T1w.nii.gz'

    # Get a list of fMRI data files in the directory
    t1_files = glob.glob(os.path.join(t1_template))
    t1_files.sort()

    shape = (192,448,512)

    # Open the output file
    with open('/mnt/workdir/DCM/tmp/check_mri_data_completeness_t1_output.txt', 'w') as f:
        # Loop through the fMRI data files
        for t1_file in t1_files:
            check_t1_completeness(t1_file,f,shape)
