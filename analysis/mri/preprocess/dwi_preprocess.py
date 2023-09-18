import os
import pandas as pd
import time
import subprocess
from nilearn import image

#%%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
subs_id = data['Participant_ID'].to_list()

fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsirecon' # look out

# Find existing subjects
exist_subjects = [file for file in os.listdir(fmriprep_dir) if 'sub-' in file]

# Find unprocessed subjects
unexist_subjects = [f for f in subs_id if f not in exist_subjects]

subject_list = [p.split('-')[-1] for p in unexist_subjects]
subject_list.sort()

valid_subjects = []
for pid in subject_list:
    dwi_exist = os.path.exists(rf'/mnt/workdir/DCM/BIDS/sub-{pid}/dwi/sub-{pid}_dir-PA_dwi.nii.gz')
    if dwi_exist:
        valid_subjects.append(pid)
    else:
        print(pid,":dwi file doesn't exist.")

# Split subjects into units to prevent memory overflow
sub_list = []
sub_set_num = 0
sub_set = ''
for i, sub in enumerate(valid_subjects):
    sub_set += sub + ' '
    sub_set_num += 1
    if sub_set_num == 30:
        sub_list.append(sub_set[:-1])
        sub_set_num = 0
        sub_set = ''
    elif i == (len(valid_subjects) - 1):  # Corrected line
        sub_list.append(sub_set[:-1])
    else:
        continue


#%%
qsiprep = 'qsiprep-docker {} {} participant --participant_label {} --output-resolution 2 --recon_input {} --recon_spec {} --fs-license-file {} --freesurfer-input {} -w {} --custom_atlases {} --nthreads 88 --gpus all'
qsiprep_without_recon = 'qsiprep-docker {} {} participant --participant_label {} --output-resolution 2 --fs-license-file {} -w {} --nthreads 88 --gpus all -v'
qsiprep_recon = 'qsiprep-docker {} {} participant --participant_label {} --output-resolution 2 --recon-only --recon_input {} --recon_spec {} --fs-license-file {} --freesurfer-input {} -w {} --custom_atlases {} --nthreads 88 --gpus all --stop-on-first-crash'
bids_dir = r'/mnt/workdir/DCM/BIDS'
out_dir = r'/mnt/workdir/DCM/BIDS/derivatives/qsiprep'
recon_input = r'/mnt/workdir/DCM/BIDS/derivatives/qsiprep'
recon_spec = '/mnt/workdir/DCM/Docs/Reference/qsiprep/mrtrix_multishell_msmt_ACT-hsvs.json'
freesurfer_license = r'/mnt/data/license.txt'
freesurfer_input = r'/mnt/workdir/DCM/BIDS/derivatives/freesurfer'
work_dir = r'/mnt/workdir/DCM/working'
path2atlas = r'/mnt/workdir/DCM/Docs/Reference/qsiprep/qsirecon_atlases'

starttime = time.time()
for subj in sub_list:
    freesurfer_license = r'/mnt/data/license.txt'
    # command = qsiprep_without_recon.format(bids_dir,out_dir,subj,recon_input,recon_spec,freesurfer_license,freesurfer_input,work_dir,path2atlas)
    # command = qsiprep_without_recon.format(bids_dir,out_dir,subj,freesurfer_license,work_dir)
    command = qsiprep_recon.format(bids_dir,out_dir,subj,recon_input,recon_spec,freesurfer_license,freesurfer_input,work_dir,path2atlas)
    print("Command:",command)
    subprocess.call(command, shell=True)

endtime = time.time()
print('总共的时间为:', round((endtime - starttime)/60/60,2), 'h')
