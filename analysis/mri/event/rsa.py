import os
import numpy as np
import pandas as pd
from analysis.mri.event.base import GAME1EV_base_spct,GAME1EV_base_spat
from analysis.mri.event.hexagon import GAME1EV_hexagon_spat,Game2EV_hexagon_spat


class GAME1_Grid_rsa_m2(GAME1EV_hexagon_spat):
    def __init__(self, behDataPath):
        GAME1EV_hexagon_spat.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def genM2ev_corr(self, trial_label):
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

    def angle_ev(self):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for index, row in self.behData.iterrows():
                onset = row['pic2_render.started'] - self.starttime
                duration = 2.5
                angle = row['angles']
                angle = round(angle % 360)
                trial_type = 'angle'+str(angle)
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for index, row in self.behData.iterrows():
                onset = row['pic2_render.started_raw'] - self.starttime
                duration = 2.5
                angle = row['angles']
                angle = round(angle % 360)
                trial_type = 'angle'+str(angle)
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
        deev = self.genDeev()
        event_data = pd.concat([m1ev,angle_ev,deev], axis=0)
        return event_data

    def correct_angle_ev(self, trial_corr):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for (index, row),isCorr in zip(self.behData.iterrows(), trial_corr):
                onset = row['pic2_render.started'] - self.starttime
                duration = 2.5
                if isCorr:
                    angle = row['angles']
                    angle = round(angle % 360)
                    trial_type = 'angle'+str(angle)
                else:
                    trial_type = 'M2_error'
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for (index, row),isCorr in zip(self.behData.iterrows(), trial_corr):
                onset = row['pic2_render.started_raw'] - self.starttime
                duration = 2.5
                if isCorr:
                    angle = row['angles']
                    angle = round(angle % 360)
                    trial_type = 'angle'+str(angle)
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
        corr_angle_ev = self.correct_angle_ev(trial_corr)
        m2ev_corr,m2ev_error = self.genM2ev_corr(trial_corr)
        deev = self.genDeev()
        event_data = pd.concat([m1ev,m2ev_corr,corr_angle_ev,deev], axis=0)
        return event_data


class GAME1EV_map_rsa_spct(GAME1EV_base_spct):
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def monsters_ev(self,trial_label):
        if self.dformat == 'trial_by_trial':
            # set m1 event
            onset = self.behData['pic1_render.started'] - self.starttime
            duration = self.behData['pic2_render.started'] - self.behData['pic1_render.started']
            ap = self.behData['pic1_ap'].to_list()
            dp = self.behData['pic1_dp'].to_list()
            m1ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic1'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m1ev['trial_type'] = pic_name
            m1ev['modulation'] = 1
            # set m2 event
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            ap = self.behData['pic2_ap'].to_list()
            dp = self.behData['pic2_dp'].to_list()
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic2'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m2ev['trial_type'] = pic_name
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            # set m1 event
            onset = self.behData['pic1_render.started_raw'] - self.starttime
            duration = self.behData['pic2_render.started_raw'] - self.behData['pic1_render.started_raw']
            ap = self.behData['pic1_ap'].to_list()
            dp = self.behData['pic1_dp'].to_list()
            m1ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic1'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m1ev['trial_type'] = pic_name
            m1ev['modulation'] = 1
            # set m2 event
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            ap = self.behData['pic2_ap'].to_list()
            dp = self.behData['pic2_dp'].to_list()
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic2'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m2ev['trial_type'] = pic_name
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        m1ev_corr = m1ev.iloc[correct_trials_index].copy()
        m2ev_corr = m2ev.iloc[correct_trials_index].copy()

        m1ev_error = m1ev.iloc[error_trials_index].copy()
        m2ev_error = m2ev.iloc[error_trials_index].copy()

        monsters_ev_corr = pd.concat([m1ev_corr, m2ev_corr], axis=0)
        monsters_ev_error = pd.concat([m1ev_error, m2ev_error], axis=0)
        monsters_ev_error['trial_type'] = 'error'

        monsters_ev_corr = monsters_ev_corr.sort_values(by=['onset'])
        monsters_ev_error = monsters_ev_error.sort_values(by=['onset'])
        return monsters_ev_corr, monsters_ev_error

    def game1ev_map_rsa(self):
        # label the correct trials and incorrect trials
        trial_label, accuracy = self.label_trial_corr()
        # generate event for each monster
        mos_ev_corr,mos_ev_error = self.monsters_ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        event_data = pd.concat([mos_ev_corr,mos_ev_error,deev_corr, deev_error], axis=0)
        return event_data


class GAME1EV_map_rsa_spat(GAME1EV_base_spat):
    def __init__(self, behDataPath):
        GAME1EV_base_spat.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def monsters_ev(self):
        if self.dformat == 'trial_by_trial':
            # set m1 event
            onset = self.behData['pic1_render.started'] - self.starttime
            duration = self.behData['pic2_render.started'] - self.behData['pic1_render.started']
            ap = self.behData['pic1_ap'].to_list()
            dp = self.behData['pic1_dp'].to_list()
            m1ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic1'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m1ev['trial_type'] = pic_name
            m1ev['modulation'] = 1
            # set m2 event
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            ap = self.behData['pic2_ap'].to_list()
            dp = self.behData['pic2_dp'].to_list()
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic2'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m2ev['trial_type'] = pic_name
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            # set m1 event
            onset = self.behData['pic1_render.started_raw'] - self.starttime
            duration = self.behData['pic2_render.started_raw'] - self.behData['pic1_render.started_raw']
            ap = self.behData['pic1_ap'].to_list()
            dp = self.behData['pic1_dp'].to_list()
            m1ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic1'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m1ev['trial_type'] = pic_name
            m1ev['modulation'] = 1
            # set m2 event
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            ap = self.behData['pic2_ap'].to_list()
            dp = self.behData['pic2_dp'].to_list()
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration,'AP':ap,'DP':dp})
            pic_path = self.behData['pic2'].to_list()
            pic_name = ['p'+path.split('/')[-1].split('.')[0] for path in pic_path]
            m2ev['trial_type'] = pic_name
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        monsters_ev = pd.concat([m1ev, m2ev], axis=0)
        monsters_ev = monsters_ev.sort_values(by=['onset'])
        return monsters_ev

    def game1ev_map_rsa_spat(self):
        # generate event for each monster
        mos_ev = self.monsters_ev()
        deev = self.genDeev()
        event_data = pd.concat([mos_ev,deev], axis=0)
        return event_data


class Game1_grid_rsa_decision(GAME1EV_hexagon_spat):
    def __init__(self, behDataPath):
        GAME1EV_hexagon_spat.__init__(self, behDataPath)
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
        m2ev = self.genM2ev()
        angle_ev = self.decision_angle_ev()
        event_data = pd.concat([m1ev,m2ev,angle_ev], axis=0)
        return event_data


class Game2_grid_rsa_m2(Game2EV_hexagon_spat):
    def __init__(self, behDataPath):
        Game2EV_hexagon_spat.__init__(self, behDataPath)
        self.behData['angles'] = round(self.behData['angles'], 1)

    def angle_ev(self):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for index, row in self.behData.iterrows():
                onset = row['testPic2.started'] - self.starttime
                duration = 2.5
                angle = row['angles']
                angle = round(angle % 360)
                trial_type = str(angle)
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for index, row in self.behData.iterrows():
                onset = row['testPic2.started_raw'] - self.starttime
                duration = 2.5
                angle = row['angles']
                angle = round(angle % 360)
                trial_type = str(angle)
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
        deev = self.genDeev()
        #response = self.response()
        event_data = pd.concat([m1ev,angle_ev,deev], axis=0)
        return event_data

    def correct_angle_ev(self, trial_corr):
        angle_ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        if self.dformat == 'trial_by_trial':
            for (index, row),isCorr in zip(self.behData.iterrows(), trial_corr):
                onset = row['testPic2.started'] - self.starttime
                duration = 2.5
                if isCorr:
                    angle = row['angles']
                    angle = round(angle % 360)
                    trial_type = 'angle'+str(angle)
                else:
                    trial_type = 'M2_error'
                angle_ev = angle_ev.append(
                    {'onset': onset, 'duration': duration, 'trial_type': trial_type, 'modulation': 1},
                    ignore_index=True)
        elif self.dformat == 'summary':
            for (index, row),isCorr in zip(self.behData.iterrows(), trial_corr):
                onset = row['testPic2.started_raw'] - self.starttime
                duration = 2.5
                if isCorr:
                    angle = row['angles']
                    angle = round(angle % 360)
                    trial_type = 'angle'+str(angle)
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
        corr_angle_ev = self.correct_angle_ev(trial_corr)
        deev = self.genDeev()
        event_data = pd.concat([m1ev,corr_angle_ev,deev], axis=0)
        return event_data


def gen_gird_rsa_event(task):
    # define subject list
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    # set folds and runs for cross validation
    ifolds = range(6, 7)
    runs = range(1,7)
    if task == 'game1':
        data = participants_data.query("game1_fmri>=0.5")
        pid = data['Participant_ID'].to_list()
        subject_list = [p.split('-')[-1] for p in pid]

        # define the template of behavioral file
        behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-game1_run-{}.csv'
        save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/grid_rsa/sub-{}/{}fold'
        event_file = 'sub-{}_task-game1_run-{}_events.tsv'
    elif task == 'game2':
        data = participants_data.query("game2_fmri>=0.5")
        pid = data['Participant_ID'].to_list()
        subject_list = [p.split('-')[-1] for p in pid]

        # define the template of behavioral file
        behav_path = r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-game2_run-{}.csv'
        save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/grid_rsa_corr_trials/sub-{}/{}fold'
        event_file = 'sub-{}_task-game2_run-{}_events.tsv'
    else:
        raise Exception("The task is wrong.")

    for sub in subject_list:
        print(sub, "started.")
        for ifold in ifolds:
            for run_id in runs:
                # generate event
                behDataPath = behav_path.format(sub, sub, run_id)
                if task == 'game1':
                    if trial_type == 'all':
                        game1_cv = GAME1_Grid_rsa_m2(behDataPath)
                        event = game1_cv.grid_rsa_ev()
                    elif trial_type == 'corr':
                        game1_cv = GAME1_Grid_rsa_m2(behDataPath)
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
    task = 'game1'
    trial_type = 'all'
    gen_gird_rsa_event(task)