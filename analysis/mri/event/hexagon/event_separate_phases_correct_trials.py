import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV
from analysis.mri.event.game2_event import Game2EV


def gen_sub_event(task_type, subjects, ifolds=range(4, 9)):
    if task_type == 'game1':
        runs = range(1, 7)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'hexagon_separate_phases_correct_trials/sub-{}/{}fold',
                      'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    elif task_type == 'game2':
        runs = range(1, 3)
        parameters = {
            'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                          r'fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
            'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                        r'hexagon_separate_phases_correct_trials/sub-{}/{}fold',
            'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")

    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))

        for ifold in ifolds:
            save_dir = parameters['save_dir'].format(task_type, subj, ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in runs:
                run_id = str(idx)
                behDataPath = parameters['behav_path'].format(subj, subj, task_type, run_id)
                if task_type == 'game1':
                    event = Game1EV(behDataPath)
                    event_data = event.game1ev(ifold)
                elif task_type == 'game2':
                    event = Game2EV(behDataPath)
                    event_data = event.game2ev_corr_trials(ifold)
                else:
                    raise Exception("The type of task is wrong.")
                tsv_save_path = join(save_dir, parameters['event_file'].format(subj, task_type, run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    task = 'game1'
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects_list = [p.split('-')[-1] for p in pid]
    gen_sub_event(task, subjects_list)
