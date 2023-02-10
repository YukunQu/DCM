import os
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV
from analysis.mri.event.game2_event import Game2EV


class Game1_grid_rsa_m2(Game1EV):
    def __init__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def angle_ev(self):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for index, row in self.behData.iterrows():
                onset = row['pic2_render.started'] - self.starttime
                duration = 2.5
                trial_type = str(row['angles'])
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for index, row in self.behData.iterrows():
                onset = row['pic2_render.started_raw'] - self.starttime
                duration = 2.5
                trial_type = str(row['angles'])
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        else:
            raise Exception("You need specify behavioral data format.")
        return angle_ev

    def grid_rsa_ev(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        angle_ev = self.angle_ev()
        deev = self.genDeev_whole_trials()
        response = self.response()
        event_data = pd.concat([m1ev,angle_ev,deev,response], axis=0)
        return event_data

    def correct_angle_ev(self,trial_valid):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for (index, row),isValid in zip(self.behData.iterrows(), trial_valid):
                onset = row['pic2_render.started'] - self.starttime
                duration = 2.5
                if isValid:
                    trial_type = str(row['angles'])
                else:
                    trial_type = 'M2_error'
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for (index, row),isValid in zip(self.behData.iterrows(), trial_valid):
                onset = row['pic2_render.started_raw'] - self.starttime
                duration = 2.5
                if isValid:
                    trial_type = str(row['angles'])
                else:
                    trial_type = 'M2_error'
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        else:
            raise Exception("You need specify behavioral data format.")
        return angle_ev

    def grid_rsa_corr_trials_ev(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        valid_angle_ev = self.correct_angle_ev(trial_corr)
        deev = self.genDeev_whole_trials()
        response = self.response()
        event_data = pd.concat([m1ev,valid_angle_ev,deev,response], axis=0)
        return event_data


class Game1_grid_rsa_decision(Game1EV):
    def __init__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def decision_angle_ev(self):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for index, row in self.behData.iterrows():
                onset = row['cue1.started'] - self.starttime
                duration = row['cue1_2.started'] - row['cue1.started']
                trial_type = str(row['angles'])
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for index, row in self.behData.iterrows():
                onset = row['cue1.started_raw'] - self.starttime
                duration = row['cue1_2.started_raw'] - row['cue1.started_raw']
                trial_type = str(row['angles'])
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        else:
            raise Exception("You need specify behavioral data format.")
        return angle_ev

    def decision_grid_rsa_ev(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        m2ev = self.genM2ev_whole_trials()
        angle_ev = self.decision_angle_ev()
        event_data = pd.concat([m1ev,m2ev,angle_ev], axis=0)
        return event_data


class Game2_grid_rsa_m2(Game2EV):
    def __init__(self, behDataPath):
        Game2EV.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def angle_ev(self):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for index, row in self.behData.iterrows():
                onset = row['testPic2.started'] - self.starttime
                duration = 2.5
                trial_type = str(row['angles'])
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for index, row in self.behData.iterrows():
                onset = row['testPic2.started_raw'] - self.starttime
                duration = 2.5
                trial_type = str(row['angles'])
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        else:
            raise Exception("You need specify behavioral data format.")
        return angle_ev

    def grid_rsa_ev(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        angle_ev = self.angle_ev()
        deev = self.genDeev_whole_trials()
        response = self.response()
        event_data = pd.concat([m1ev,angle_ev,deev,response], axis=0)
        return event_data

    def correct_angle_ev(self,trial_valid):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for (index, row),isValid in zip(self.behData.iterrows(), trial_valid):
                onset = row['testPic2.started'] - self.starttime
                duration = 2.5
                if isValid:
                    trial_type = str(row['angles'])
                else:
                    trial_type = 'M2_error'
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for (index, row),isValid in zip(self.behData.iterrows(), trial_valid):
                onset = row['testPic2.started_raw'] - self.starttime
                duration = 2.5
                if isValid:
                    trial_type = str(row['angles'])
                else:
                    trial_type = 'M2_error'
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        else:
            raise Exception("You need specify behavioral data format.")
        return angle_ev

    def grid_rsa_corr_trials_ev(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        valid_angle_ev = self.correct_angle_ev(trial_corr)
        deev = self.genDeev_whole_trials()
        event_data = pd.concat([m1ev,valid_angle_ev,deev], axis=0)
        return event_data


def gen_gird_rsa_event(task):
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')

    if task == 'game1':
        data = participants_data.query("game1_fmri>=0.5")
        pid = data['Participant_ID'].to_list()
        subject_list = [p.split('-')[-1] for p in pid]

        # define the template of behavioral file
        behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
        save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/grid_rsa_corr_trials/sub-{}/{}fold'
        event_file = 'sub-{}_task-game1_run-{}_events.tsv'

        # set folds and runs for cross validation
        ifolds = range(6, 7)
        runs = range(1, 7)
    elif task == 'game2':
        data = participants_data.query("game2_fmri>0.5")
        pid = data['Participant_ID'].to_list()
        subject_list = [p.split('-')[-1] for p in pid]

        # define the template of behavioral file
        behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-game2_run-{}.csv'
        save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/grid_rsa_corr_trials/sub-{}/{}fold'
        event_file = 'sub-{}_task-game2_run-{}_events.tsv'

        # set folds and runs for cross validation
        ifolds = range(6, 7)
        runs = range(1, 3)
    else:
        raise Exception("The task is wrong.")

    for sub in subject_list:
        print(sub, "started.")
        for ifold in ifolds:
            for run_id in runs:
                # generate event
                behDataPath = behav_path.format(sub, sub, run_id)
                if task == 'game1':
                    game1_cv = Game1_grid_rsa_m2(behDataPath)
                    event = game1_cv.grid_rsa_corr_trials_ev()
                elif task == 'game2':
                    game2_cv = Game2_grid_rsa_m2(behDataPath)
                    event = game2_cv.grid_rsa_corr_trials_ev()
                else:
                    raise Exception("The task is wrong.")
                # save
                out_dir = save_dir.format(sub, ifold)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                tsv_save_path = os.path.join(out_dir, event_file.format(sub, run_id))
                event.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    task = 'game2'
    gen_gird_rsa_event(task)
