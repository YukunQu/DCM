import os
import pandas as pd
import subprocess


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()

hemi = 'lh'
roi = 'mPFC'
segvol = f'/mnt/workdir/DCM/Docs/Mask/VMPFC/{hemi}.fsaverage.{roi}.mgh'
outdir = f'/mnt/workdir/DCM/BIDS/derivatives/freesurfer_stats/volume/{hemi}.{roi}'
if not os.path.exists(outdir):
    os.makedirs(outdir)
stats = 'volume'

for subjid in subject_list:
    subj_surf_dir = f'/mnt/workdir/DCM/BIDS/derivatives/freesurfer/{subjid}/surf'

    outpath = f'{outdir}/segstats-{subjid}.txt'

    cmd1 = f'mri_surf2surf --s {subjid} --trgsubject fsaverage --hemi {hemi} --sval {subj_surf_dir}/{hemi}.{stats} --tval {subj_surf_dir}/{hemi}.{stats}.fsaverage.mgh'
    print("Command:",cmd1)
    subprocess.call(cmd1, shell=True)

    if stats == 'thickness':
        cmd2 = f'mri_segstats --seg {segvol} --in {subj_surf_dir}/{hemi}.{stats}.fsaverage.mgh --sum {outpath}'
    elif stats == 'volume':
        cmd2 = f'mri_segstats --seg {segvol} --in {subj_surf_dir}/{hemi}.{stats}.fsaverage.mgh --sum {outpath} --accumulate'
    print("Command:", cmd2)
    subprocess.call(cmd2, shell=True)
    print(subjid,'finsihed.')

#%%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()