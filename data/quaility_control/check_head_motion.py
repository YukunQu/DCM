import os
import pandas as pd
from os.path import join as pjoin

"""Check the head motion for each run of each subject."""

# get subject list
fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_ica'
file_list = os.listdir(fmriprep_dir)
sub_list = []
for f in file_list:
    if ('sub' in f) and ('.html' not in f):
        sub_list.append(f)
sub_list.sort()

subs_hm = pd.DataFrame(columns=['Participant_ID','task','run','FD_mean'])
for sub in sub_list:
    print(f"--------------{sub} start------------")
    # read head motion for each subject
    for run_id in range(1,7):
        # game1
        confound_file = pjoin(fmriprep_dir,f'{sub}/func/{sub}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv')
        if os.path.exists(confound_file):
            confound_data = pd.read_csv(confound_file,sep='\t')
            fd = confound_data['framewise_displacement'].mean()
        else:
            print("The",sub,f"didn't have game1 run-{run_id}")
            fd = 999
        subs_hm = subs_hm.append({'Participant_ID':sub,'task':'game1','run':run_id,'FD_mean':fd},ignore_index=True)

    for run_id in range(1,3):
        # game2
        confound_file = pjoin(fmriprep_dir,f'{sub}/func/{sub}_task-game2_run-{run_id}_desc-confounds_timeseries.tsv')
        if os.path.exists(confound_file):
            confound_data = pd.read_csv(confound_file,sep='\t')
            fd = confound_data['framewise_displacement'].mean()
        else:
            print("The",sub,f"didn't have game2 run-{run_id}")
            fd = 999
        subs_hm = subs_hm.append({'Participant_ID':sub,'task':'game2','run':run_id,'FD_mean':fd},ignore_index=True)

    for run_id in range(1,3):
        # rest
        confound_file = pjoin(fmriprep_dir,f'{sub}/func/{sub}_task-rest_run-{run_id}_desc-confounds_timeseries.tsv')
        if os.path.exists(confound_file):
            confound_data = pd.read_csv(confound_file,sep='\t')
            fd = confound_data['framewise_displacement'].mean()
        else:
            print("The",sub,f"didn't have rest run-{run_id}")
            fd = 999
        subs_hm = subs_hm.append({'Participant_ID':sub,'task':'game2','run':run_id,'FD_mean':fd},ignore_index=True)

#%%
bad_subs_hm = subs_hm.query("(FD_mean>0.2)and(FD_mean<999)")
bad_subs = set(bad_subs_hm['Participant_ID'])

participant_df = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv',sep='\t')
pid  = participant_df['Participant_ID']
pid = [p.replace("_",'-') for p in pid]
participant_df['Participant_ID'] = pid

bad_sub_summary = pd.DataFrame(columns=['Participant_ID','Name','Age',
                                        'Total number of bad runs for game1',
                                        'Total number of bad runs for game2'])
for sub in bad_subs:
    sub_info = participant_df.query(f"Participant_ID=='{sub}'")
    name = sub_info['Name'].values[0]
    age = sub_info['Age'].values[0]
    sub_fd = bad_subs_hm.query(f"Participant_ID=='{sub}'")['FD_mean'].mean()
    game1_bad_run_number = len(bad_subs_hm.query(f"(Participant_ID=='{sub}')and(task=='game1')"))
    game2_bad_run_number = len(bad_subs_hm.query(f"(Participant_ID=='{sub}')and(task=='game2')"))

    bad_sub_summary = bad_sub_summary.append({'Participant_ID':sub,'Name':name,'Age':age,
                                              'Total number of bad runs for game1':game1_bad_run_number,
                                              'Total number of bad runs for game2':game2_bad_run_number,
                                              'FD_mean':sub_fd},ignore_index=True)