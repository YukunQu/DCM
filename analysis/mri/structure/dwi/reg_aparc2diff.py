import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
# the scrpit to register aparc+aseg to diffusion space and extract the corresponding metrics from ROIs parallelly

"""Example for registration of aparc+aseg to diffusion space
bbregister --s sub-017 --mov /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/sub-017_dir-PA_space-T1w_dwiref.nii.gz  --reg /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/register.dat --dti

mri_vol2vol --mov /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/sub-017_dir-PA_space-T1w_dwiref.nii.gz --targ /mnt/workdir/DCM/BIDS/derivatives/freesurfer/sub-017/mri/aparc+aseg.mgz --inv --interp nearest --o /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/aparc+aseg2diff.mgz --reg /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/register.dat --no-save-reg

mri_segstats --seg /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/aparc+aseg2diff.mgz --ctab $FREESURFER_HOME/FreeSurferColorLUT.txt --i /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/sub-017_dwi_FA.nii.gz --sum /mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep/sub-017/dwi/fa.stats
"""

# Define a function to run the commands for a single subject
def process_subject(sub):
    print(sub)
    # define all the file paths
    dwiref_file = f"{qsiprep_dir}/{sub}/dwi/{sub}_dir-PA_space-T1w_dwiref.nii.gz"
    register_dat_file = f"{qsiprep_dir}/{sub}/dwi/register.dat"
    aparc_file = f"/mnt/workdir/DCM/BIDS/derivatives/freesurfer/{sub}/mri/aparc+aseg.mgz"
    aparc2diff_file = f"{qsiprep_dir}/{sub}/dwi/aparc+aseg2diff.mgz"
    fa_file = f"{qsiprep_dir}/{sub}/dwi/{sub}_dwi_FA.nii.gz"
    md_file = f"{qsiprep_dir}/{sub}/dwi/{sub}_dwi_ADC.nii.gz"
    # fa_file_masked = f"{qsiprep_dir}/{sub}/dwi/{sub}_fa-masked.mgz"
    # md_file_masked = f"{qsiprep_dir}/{sub}/dwi/{sub}_md-masked.mgz"
    fastats_file = f"{qsiprep_dir}/{sub}/dwi/fa_close1.stats"
    mdstats_file = f"{qsiprep_dir}/{sub}/dwi/md_close1.stats"
    color_lut_file = f"{freesurfer_home}/FreeSurferColorLUT.txt"

    # check if all files exist
    for file in [dwiref_file, aparc_file, fa_file, md_file]:
        if not os.path.isfile(file):
            raise Exception(f"File does not exist: {file}")

    # run the commands
    subprocess.check_call(['bbregister',  '--s', sub, '--mov', dwiref_file, '--reg', register_dat_file, '--dti'])
    subprocess.check_call(['mri_vol2vol', '--mov', dwiref_file, '--targ', aparc_file, '--inv', '--interp', 'nearest', '--o', aparc2diff_file, '--reg', register_dat_file, '--no-save-reg'])
    # subprocess.check_call(['mri_mask', fa_file, aparc2diff_file, fa_file_masked])
    # subprocess.check_call(['mri_mask', md_file, aparc2diff_file, md_file_masked])
    subprocess.check_call(['mri_segstats','--seg', aparc2diff_file, '--ctab', color_lut_file, '--i', fa_file, '--sum', fastats_file])
    subprocess.check_call(['mri_segstats','--seg', aparc2diff_file, '--ctab', color_lut_file, '--i', md_file, '--sum', mdstats_file])

# load subjects list
qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'

# check if directory exists
if not os.path.isdir(qsiprep_dir):
    raise Exception(f"Directory does not exist: {qsiprep_dir}")

# get sublist
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]

if not sub_list:
    raise Exception("No subjects found")

sub_list.sort()

# check if FREESURFER_HOME environment variable is set
if 'FREESURFER_HOME' not in os.environ:
    raise Exception("FREESURFER_HOME environment variable is not set")
freesurfer_home = os.environ['FREESURFER_HOME']

# Create a process pool and run the commands for all subjects in parallel
with ProcessPoolExecutor() as executor:
    executor.map(process_subject, sub_list)