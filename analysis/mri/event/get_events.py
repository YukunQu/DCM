import os
import pandas as pd
from os.path import join


ifolds = [6]
task = 'game1'
glm_type = 'hexagon_distance_spct'
template = {'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/' + glm_type + '/sub-{}/{}fold',
            'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}

runs = range(1, 7) if task == 'game1' else range(1, 3)

participants_data = pd.read_csv('/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
subjects = participants_data.query(f'{task}_fmri>=0.5')['Participant_ID'].str.split('-').str[-1].str.zfill(3)

for subj in subjects:
    print(f'----sub-{subj}----')

    for ifold in ifolds:
        save_dir = template['save_dir'].format(task, subj, ifold)
        os.makedirs(save_dir, exist_ok=True)

        for idx in runs:
            run_id = str(idx)
            behav_path = template['behav_path'].format(subj, subj, task, run_id)
            event_data = getattr(eval(f'{task.upper()}EV_{glm_type}')(behav_path), f'{task.lower()}ev_{glm_type}')()

            tsv_save_path = join(save_dir, template['event_file'].format(subj, task, run_id))
            event_data.to_csv(tsv_save_path, sep='\t', index=False)