import os
from os.path import join
import pandas as pd
from analysis.mri.event.game1_event import Game1EV
from analysis.mri.event.game2_event import Game2EV


class Game1EV_hexagon_spct(Game1EV):
    """ event separate phases correct trials"""
    def __int__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)

    def genM2ev(self, trial_label):
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

        assert len(m2ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        m2ev_corr = m2ev.iloc[correct_trials_index].copy()
        m2ev_error = m2ev.iloc[error_trials_index].copy()
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr,m2ev_error

    def genDeev(self, trial_label):
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

        assert len(deev) == len(
            trial_label), "The number of trial label didn't not  same as the number of event-decision."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        deev_corr = deev.iloc[correct_trials_index].copy()
        deev_error = deev.iloc[error_trials_index].copy()
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_corr, deev_error

    def game1ev_hexagon_spct(self, ifold):
        # base regressors
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # paramertric modulation regressors
        m2_pmod_sin, m2_pmod_cos = self.genpm(m2ev_corr, ifold)
        decision_pmod_sin, decision_pmod_cos = self.genpm(deev_corr, ifold)
        sin = pd.concat([m2_pmod_sin,decision_pmod_sin],axis=0).sort_values('onset', ignore_index=True)
        cos = pd.concat([m2_pmod_cos,decision_pmod_cos],axis=0).sort_values('onset', ignore_index=True)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error,deev_corr, deev_error,
                                sin, cos], axis=0)
        return event_data


class Game2EV_hexagon_spct(Game2EV):
    def __int__(self, behDataPath):
        Game2EV.__init__(self, behDataPath)

    def genM2ev(self, trial_label):
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

        assert len(m2ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        m2ev_corr = m2ev.iloc[correct_trials_index].copy()
        m2ev_error = m2ev.iloc[error_trials_index].copy()
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr, m2ev_error

    def genDeev(self, trial_label):
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

        assert len(deev) == len(trial_label), "The number of trial label didn't not  same as the number of " \
                                              "event-decision. "

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        deev_corr = deev.iloc[correct_trials_index].copy()
        deev_error = deev.iloc[error_trials_index].copy()
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset', ignore_index=True)
        deev_error = deev_error.sort_values('onset', ignore_index=True)
        return deev_corr, deev_error

    def game2ev_hexagon_spct(self, ifold):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        pmod_sin, pmod_cos = self.genpm(m2ev_corr, ifold)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_sin, pmod_cos], axis=0)
        return event_data


def gen_sub_event(task_type, subjects, ifolds=range(4, 9)):
    if task_type == 'game1':
        runs = range(1, 7)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'hexagon_spct/sub-{}/{}fold',
                      'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}
    elif task_type == 'game2':
        runs = range(1, 3)
        parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                    r'fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                      'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                                  r'hexagon_spct/sub-{}/{}fold',
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
                    event = Game1EV_hexagon_spct(behDataPath)
                    event_data = event.game1ev_hexagon_spct(ifold)
                elif task_type == 'game2':
                    event = Game2EV_hexagon_spct(behDataPath)
                    event_data = event.game2ev_hexagon_spct(ifold)
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
    subjects_list = ['209','250']
    gen_sub_event(task, subjects_list)
