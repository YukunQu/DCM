import os
import pandas as pd
from os.path import join
from analysis.mri.event.base import GAME1EV_base_spct,GAME2EV_base_spct
from analysis.mri.event.hexagon import GAME1EV_hexagon_spct,GAME2EV_hexagon_spct,GAME2EV_hexagon_center_spct
from analysis.mri.event.distance import GAME1EV_2distance_spct,GAME1EV_distance_spct,GAME2EV_2distance_spct,GAME2EV_distance_spct,GAME2EV_distance_center_spct
from analysis.mri.event.value import GAME1EV_value_spct,GAME2EV_value_spct


ifolds = range(6,7)
task = 'game1'
glm_type = 'base_spct'
drop_stalemate= False
print(glm_type)
template = {'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'+glm_type+'/sub-{}/{}fold',
            'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
if task == 'game1':
    runs = range(1, 7)
    template['behav_path'] = '/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/' \
                             'fmri_task-game1/sub-{}_task-{}_run-{}.csv'
elif task == 'game2':
    runs = range(1, 3)
    template['behav_path'] = '/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/' \
                             'fmri_task-game2-test/sub-{}_task-{}_run-{}.csv'
else:
    raise Exception("Task is not supported.")

participants_data = pd.read_csv('/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
subjects = participants_data.query(f'{task}_fmri>=0.5')['Participant_ID'].str.split('-').str[-1].str.zfill(3)
#%%
for subj in subjects:
    print(f'----sub-{subj}----')

    for ifold in ifolds:
        save_dir = template['save_dir'].format(task, subj, ifold)
        os.makedirs(save_dir, exist_ok=True)

        for idx in runs:
            run_id = str(idx)
            behav_path = template['behav_path'].format(subj, subj, task, run_id)
            if 'hexagon' in glm_type:
                event_data = getattr(eval(f'{task.upper()}EV_{glm_type}')(behav_path), f'{task.lower()}ev_{glm_type}')(ifold,drop_stalemate)
            else:
                if drop_stalemate:
                    event_data = getattr(eval(f'{task.upper()}EV_{glm_type}')(behav_path), f'{task.lower()}ev_{glm_type}')(True)
                else:
                    event_data = getattr(eval(f'{task.upper()}EV_{glm_type}')(behav_path), f'{task.lower()}ev_{glm_type}')()
            tsv_save_path = join(save_dir, template['event_file'].format(subj, task, run_id))
            event_data.to_csv(tsv_save_path, sep='\t', index=False)