import pandas as pd
from analysis.mri.Whole_brain_analysis.firstLevel import firstLevel_noPhi_separate


# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game2_fmri==1')
pid = data['Participant_ID'].to_list()
subject_list = [p.split('_')[-1] for p in pid]

# input files
configs = {'data_root': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_ica',
           'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
           'task':'game2',
           'glm_type': 'separate_hexagon',
           'func_name':'sub-{subj_id}_task-game2_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
           'event_name':'sub-{subj_id}_task-game2_run-{run_id}_events.tsv',
           'regressor_name':'sub-{subj_id}_task-game2_run-{run_id}_desc-confounds_timeseries.tsv'}

set_id = 'all'
runs = [1,2,3,4,5,6]
ifold = '6fold'
firstLevel_noPhi_separate(subject_list, set_id, runs, ifold, configs)