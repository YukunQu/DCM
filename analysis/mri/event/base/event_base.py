import os
import pandas as pd
from os.path import join
from analysis.mri.event.hexagon.event_hexagon_spct import Game1EV_hexagon_spct,Game2EV_hexagon_spct


class Game1EV_base_spct(Game1EV_hexagon_spct):
    def __init__(self,behDataPath):
        Game1EV_hexagon_spct.__init__(self,behDataPath)

    def game1ev_base_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error], axis=0)
        return event_data


def gen_sub_event(task_type, subjects, ifolds=range(4, 9)):
    if task_type == 'game1':
        runs = range(1, 7)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'base_spct/sub-{}/{}fold',
                      'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    elif task_type == 'game2':
        runs = range(1, 3)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'base_spct/sub-{}/{}fold',
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
                    event = Game1EV_base_spct(behDataPath)
                    event_data = event.game1ev_base_spct()
                elif task_type == 'game2':
                    pass
                    #event = Game2EV_hexagon_spct(behDataPath)
                    #event_data = event.game2ev_hexagon_spct(ifold)
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
