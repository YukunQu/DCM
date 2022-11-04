import os
import pandas as pd
from analysis.mri.zscore_nii import zscore_nii
from analysis.mri.voxel_wise.firstLevel import firstLevel_noPhi

# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
data = data.query("(game1_acc>=0.8)and(Age>=18)") # high performance subjects only
pid = data['Participant_ID'].to_list()

# check the existence of preprocessing file
fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep'
preprocess_subs = os.listdir(fmriprep_dir)
preprocess_subs = [p for p in preprocess_subs if ('sub-'in p) and ('html' not in p)]
for p in pid:
    if p not in preprocess_subs:
        print(f"The {p} didn't have preprocess files.")

# configure parameters
configs = {'data_root': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep',
           'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
           'task':'game1',
           'glm_type': 'separate_hexagon_2phases_correct_trials',  # look out
           'func_name':'sub-{subj_id}_task-game1_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth8_cleaned.nii.gz',
           'event_name':'sub-{subj_id}_task-game1_run-{run_id}_events.tsv',
           'regressor_name':'sub-{subj_id}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv'}

# filter the subjects who exist.
target_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/{}/Setall/6fold'.format(configs['glm_type'])
if os.path.exists(target_dir):
    already_sub = os.listdir(target_dir)
    subject_list = [p.split('-')[-1] for p in pid if p not in already_sub]
else:
    subject_list = [p.split('-')[-1] for p in pid]
print("{} subjects are ready.".format(len(subject_list)))

set_id = 'all'
runs = [1,2,3,4,5,6]
ifold = '6fold'
firstLevel_noPhi(subject_list, set_id, runs, ifold, configs)

# zscore 1st level
cmap_dir = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/{configs["task"]}/{configs["glm_type"]}/Set{set_id}'
data_dir = os.path.join(cmap_dir,ifold)
sub_list = os.listdir(data_dir)
sub_list.sort()
for sub in sub_list:
    sub_cmap_dir = os.path.join(data_dir,sub)
    cmap_list = os.listdir(sub_cmap_dir)
    for cmap in cmap_list:
        if 'spm' in cmap:
            zscore_nii(sub_cmap_dir,cmap, 'Z')
    print("The cmap of",sub,'was zscored.')