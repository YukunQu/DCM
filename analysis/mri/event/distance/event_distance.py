import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.hexagon.event_hexagon_spat import Game1EV_hexagon_spat, Game2EV_hexagon_spat
from analysis.mri.event.hexagon.event_hexagon_spct import Game1EV_hexagon_spct, Game2EV_hexagon_spct


class Game1EV_distance_spat(Game1EV_hexagon_spat):
    """Game1 distance modulation on separate phases for all trials"""
    def __init__(self, behDataPath):
        Game1EV_hexagon_spat.__init__(self, behDataPath)

    def genpm_distance_spat(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")

        pmev = pmev.sort_values('onset', ignore_index=True)
        return pmev

    def game1ev_distance_spat(self):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_distance = self.genpm_distance_spat()
        event_data = pd.concat([m1ev, m2ev, deev, pmod_distance], axis=0)
        return event_data


class Game1EV_distance_spct(Game1EV_hexagon_spct):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        Game1EV_hexagon_spct.__init__(self, behDataPath)

    def genpm_distance_spct(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")

        assert len(pmev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        pmev_corr = pmev.iloc[correct_trials_index].copy()
        pmev_corr = pmev_corr.sort_values('onset', ignore_index=True)
        return pmev_corr

    def game1ev_distance_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        pmev_corr = self.genpm_distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmev_corr], axis=0)
        return event_data


class Game1EV_hexagon_distance_spat(Game1EV_distance_spat):
    def __init__(self, behDataPath):
        Game1EV_distance_spat.__init__(self, behDataPath)

    def game1ev_hexagon_distance_spat(self):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_distance = self.genpm_distance_spat()
        pmod_sin, pmod_cos = self.genpm(m2ev, ifold)
        event_data = pd.concat([m1ev, m2ev, deev,pmod_sin, pmod_cos,pmod_distance], axis=0)
        return event_data


class Game1EV_hexagon_distance_spct(Game1EV_distance_spct):
    def __init__(self, behDataPath):
        Game1EV_distance_spct.__init__(self, behDataPath)

    def game1ev_hexagon_distance_spct(self):
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

        pmev_corr = self.genpm_distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                sin, cos, pmev_corr], axis=0)
        return event_data


class Game2EV_distance_spat(Game2EV_hexagon_spat):
    """Game2 distance modulation for whole_trials"""

    def __init__(self, behDataPath):
        Game2EV_hexagon_spat.__init__(self, behDataPath)

    def genpm_distance_spat(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")
        return pmev

    def game2ev_distance_spat(self):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_distance = self.genpm_distance_spat()

        event_data = pd.concat([m1ev, m2ev, deev, pmod_distance], axis=0)
        return event_data


class Game2EV_distance_spct(Game2EV_hexagon_spct):
    def __init__(self, behDataPath):
        Game2EV_hexagon_spct.__init__(self, behDataPath)

    def genpm_distance_spct(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")

        assert len(pmev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        pmev_corr = pmev.iloc[correct_trials_index].copy()
        pmev_corr = pmev_corr.sort_values('onset', ignore_index=True)
        return pmev_corr

    def game2ev_distance_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        pmod_distance = self.genpm_distance_spct(trial_label)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_distance], axis=0)
        return event_data


if __name__ == "__main__":
    # set configure
    ifolds = [4,5,6,7,8]  # only 6 have meaning for distance
    task = 'game1'
    glm_type = 'hexagon_distance_spct'
    template = {
        'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'+glm_type+'/sub-{}/{}fold',
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

    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]
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
                    if glm_type == 'distance_spct':
                        event = Game1EV_distance_spct(behDataPath)
                        event_data = event.game1ev_distance_spct()
                    elif glm_type == 'distance_spat':
                        event = Game1EV_distance_spat(behDataPath)
                        event_data = event.game1ev_distance_spat()
                    elif glm_type == 'hexagon_distance_spct':
                        event = Game1EV_hexagon_distance_spct(behDataPath)
                        event_data = event.game1ev_hexagon_distance_spct()
                    elif glm_type == 'hexagon_distance_spat':
                        event = Game1EV_hexagon_distance_spat(behDataPath)
                        event_data = event.game1ev_hexagon_distance_spat()
                else:
                    if glm_type == 'distance_spct':
                        event = Game2EV_distance_spct(behDataPath)
                        event_data = event.game2ev_distance_spct()
                    elif glm_type == 'distance_spat':
                        event = Game2EV_distance_spat(behDataPath)
                        event_data = event.game2ev_distance_spat()
                tsv_save_path = join(save_dir, template['event_file'].format(subj, task, run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)