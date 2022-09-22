import pandas as pd
from analysis.mri.Whole_brain_analysis.firstLevel import firstLevel_noPhi_separate


# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')
sub_type = 'hp'
hp_data = data.query('game1_acc>=0.80')
subject_list = [p.split('_')[-1] for p in hp_data['Participant_ID'].to_list()]
print("High performance:",len(hp_data),"({} adult)".format(len(hp_data.query('Age>18'))))

# input files
configs = {'data_root': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_ica',
           'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
           'task':'game1',
           'glm_type': 'separate_hexagon',
           'func_name':'sub-{subj_id}_task-game1_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
           'event_name':'sub-{subj_id}_task-game1_run-{run_id}_events.tsv',
           'regressor_name':'sub-{subj_id}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv'}

# split k training set
training_sets = {1: [1, 2, 3] #######
                 }
for set_id,runs in training_sets.items():
    for i in range(4, 9):
        ifold = str(i) + 'fold'
        firstLevel_noPhi_separate(subject_list, set_id, runs, ifold, configs)