import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV
from analysis.mri.event.game2_event import Game2EV


class Game1EV_hexagon_spat(Game1EV):
    """ game1 event for separate phases all trials"""
    def __int__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)

    def genM2ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m2ev = m2ev.sort_values('onset', ignore_index=True)
        return m2ev

    def genDeev(self):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        deev = deev.sort_values('onset', ignore_index=True)
        return deev

    def genpm(self, ev, ifold):
        # ev is reference event of one trial type to provide angle of each trial
        angle = ev['angle']
        pmod_sin = ev.copy()
        pmod_cos = ev.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold * angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold * angle))
        return pmod_sin, pmod_cos

    def game1ev_hexagon_spat(self, ifold):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_sin, pmod_cos = self.genpm(m2ev, ifold)

        event_data = pd.concat([m1ev, m2ev, deev, pmod_sin, pmod_cos], axis=0)
        return event_data


class Game2EV_hexagon_spat(Game2EV):
    """ game2 event for separate phases all trials"""
    def __int__(self, behDataPath):
        Game2EV.__init__(self, behDataPath)

    def genM2ev(self):
        # generate M2 trials event
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m2ev = m2ev.sort_values('onset', ignore_index=True)
        return m2ev

    def genDeev(self):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        deev = deev.sort_values('onset', ignore_index=True)
        return deev

    def game2ev_hexagon_spat(self, ifold):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_sin, pmod_cos = self.genpm(m2ev, ifold)
        event_data = pd.concat([m1ev, m2ev, deev, pmod_sin, pmod_cos], axis=0)
        return event_data


def gen_sub_event(task, subjects, ifolds=range(4, 9)):
    if task == 'game1':
        runs = range(1, 7)
        template = {
            'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
            'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/hexagon_spat/sub-{}/{}fold',
            'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    elif task == 'game2':
        runs = range(1, 3)
        template = {
            'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
            'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/hexagon_spat/sub-{}/{}fold',
            'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")

    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))

        for ifold in ifolds:
            save_dir = template['save_dir'].format(task, subj, ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in runs:
                run_id = str(idx)
                behDataPath = template['behav_path'].format(subj, subj, task, run_id)
                if task == 'game1':
                    event = Game1EV_hexagon_spat(behDataPath)
                    event_data = event.game1ev_hexagon_spat(ifold)
                elif task == 'game2':
                    event = Game2EV_hexagon_spat(behDataPath)
                    event_data = event.game2ev_hexagon_spat(ifold)
                else:
                    raise Exception("The type of task is wrong.")
                tsv_save_path = join(save_dir, template['event_file'].format(subj, task, run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    task = 'game2'
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]
    gen_sub_event(task, subjects)
