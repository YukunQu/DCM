import os
import shutil
import numpy as np
import pandas as pd

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game2_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()
subject_list.remove('sub-010')
subject_list = [s.split('-')[-1] for s in subject_list]
#subject_list = subject_list[:100]
subject_list = subject_list[100:]
#%%
simg_in_template = os.path.join('/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game2/separate_hexagon_2phases_correct_trials/Setall/6fold'
                             '/work_1st/_subj_id_{}/smooth',
                             'ssub-{}_task-game2_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii')

simg_out_template = os.path.join('/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{}/func/',
                                 'sub-{}_task-game2_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth8.nii')

# move T1w to fsl directory
for sub_id in subject_list:
    for run_id in range(1,3):
        simg_in = simg_in_template.format(sub_id,sub_id,run_id)
        simg_out = simg_out_template.format(sub_id,sub_id,run_id)
        shutil.copy(simg_in,simg_out)
        print(simg_out,'finished.')
