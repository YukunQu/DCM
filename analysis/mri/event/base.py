import os
import numpy as np
import pandas as pd
from os.path import join


class GAME1EV(object):
    """The father class of Game1"""
    def __init__(self, behDataPath):
        self.dformat = None
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.starttime = self.cal_start_time()

    def game1_dformat(self):
        columns = self.behData.columns
        if 'fix_start_cue.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            raise Exception("You need specify behavioral data format.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fix_start_cue.started'].min()
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min() - 1
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime

    def label_trial_corr(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['resp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['resp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            raise Exception("You need specify behavioral data format.")

        trial_corr = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                else:
                    correctAns = 2
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                else:
                    correctAns = 1
            else:
                raise Exception("None of rule have been found in the file.")
            if (keyResp == 'None') or (keyResp is None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData), 3)
        return trial_corr, accuracy

    def label_trials_stalemate(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['resp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['resp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            raise Exception("You need specify behavioral data format.")

        stalemate_trials = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
            else:
                raise Exception("None of rule have been found in the file.")
            if fight_result == 0:
                # add index of the row
                stalemate_trials.append(1)
            else:
                stalemate_trials.append(0)
        self.behData['stalemate'] = stalemate_trials
        return stalemate_trials

    @staticmethod
    def drop_sm(ev):
        # drop stalemate trials
        ev = ev.query('stalemate == 0')
        return ev

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic1_render.started'] - self.starttime
            duration = self.behData['pic2_render.started'] - self.behData['pic1_render.started']
            angle = self.behData['angles']
            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic1_render.started_raw'] - self.starttime
            duration = self.behData['pic2_render.started_raw'] - self.behData['pic1_render.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset', ignore_index=True)
        return m1ev

    def response(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            onsets = []
            duration = 0
            angles = []
            for index, row in self.behData.iterrows():
                onset = row['resp.started'] - self.starttime
                rt = row['resp.rt']
                if rt != 'None':
                    onsets.append(onset + rt)
                    angles.append(row['angles'])
            pbev = pd.DataFrame({'onset': onsets, 'duration': duration, 'angle': angles})
            pbev['trial_type'] = 'response'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onsets = []
            duration = 0
            angles = []
            for index, row in self.behData.iterrows():
                onset = row['resp.started_raw'] - self.starttime
                rt = row['resp.rt_raw']
                if rt != 'None':
                    onsets.append(onset + rt)
                    angles.append(row['angles'])
            pbev = pd.DataFrame({'onset': onsets, 'duration': duration, 'angle': angles})
            pbev['trial_type'] = 'response'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        pbev = pbev.sort_values('onset', ignore_index=True)
        return pbev


class GAME2EV(object):
    """The father class of Game2"""
    def __init__(self, behDataPath):
        self.dformat = None
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.starttime = self.cal_start_time()

    def game1_dformat(self):
        columns = self.behData.columns
        if 'fixation.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            raise Exception("You need specify behavioral data format.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fixation.started'].min()
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min()
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime

    def label_trial_corr(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['dResp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['dResp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            raise Exception("You need specify behavioral data format.")

        trial_corr = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                else:
                    correctAns = 2
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                else:
                    correctAns = 1
            else:
                raise Exception("None of rule have been found in the file.")
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData), 3)
        return trial_corr, accuracy

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic1.started'] - self.starttime
            duration = self.behData['testPic2.started'] - self.behData['testPic1.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic1.started_raw'] - self.starttime
            duration = self.behData['testPic2.started_raw'] - self.behData['testPic1.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset',ignore_index=True)
        return m1ev

    def response(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            onsets = []
            duration = 0
            angles = []
            for index, row in self.behData.iterrows():
                onset = row['dResp.started'] - self.starttime
                rt = row['dResp.rt']
                if rt != 'None':
                    onsets.append(onset + rt)
                    angles.append(row['angles'])
            pbev = pd.DataFrame({'onset': onsets, 'duration': duration, 'angle': angles})
            pbev['trial_type'] = 'response'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onsets = []
            duration = 0
            angles = []
            for index, row in self.behData.iterrows():
                onset = row['dResp.started_raw'] - self.starttime
                rt = row['dResp.rt_raw']
                if rt != 'None':
                    onsets.append(onset + rt)
                    angles.append(row['angles'])
            pbev = pd.DataFrame({'onset': onsets, 'duration': duration, 'angle': angles})
            pbev['trial_type'] = 'response'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        pbev = pbev.sort_values('onset', ignore_index=True)
        return pbev


class GAME1EV_base_spct(GAME1EV):
    # event of base GLM
    def __init__(self,behDataPath):
        GAME1EV.__init__(self,behDataPath)

    def genM2ev(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
            if 'stalemate' in self.behData.columns:
                m2ev['stalemate'] = self.behData['stalemate']
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
            if 'stalemate' in self.behData.columns:
                m2ev['stalemate'] = self.behData['stalemate']
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
            if 'stalemate' in self.behData.columns:
                deev['stalemate'] = self.behData['stalemate']
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
            if 'stalemate' in self.behData.columns:
                deev['stalemate'] = self.behData['stalemate']
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

    def game1ev_base_spct(self):
        m1ev = self.genM1ev()
        self.label_trials_stalemate()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        # concat all events
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error], axis=0)
        return event_data


class GAME2EV_base_spct(GAME2EV):
    def __int__(self, behDataPath):
        GAME2EV.__init__(self, behDataPath)

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

    def game2ev_base_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        # concat all events
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error], axis=0)
        return event_data


class GAME1EV_base_spat(GAME1EV):
    # event of base GLM
    def __init__(self,behDataPath):
        GAME1EV.__init__(self,behDataPath)

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

    def game1ev_base_spat(self):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()

        event_data = pd.concat([m1ev, m2ev, deev], axis=0)
        return event_data


class GAME2EV_base_spat(GAME2EV):
    """ game2 event for separate phases all trials"""
    def __int__(self, behDataPath):
        GAME2EV.__init__(self, behDataPath)

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
        event_data = pd.concat([m1ev, m2ev, deev], axis=0)
        return event_data