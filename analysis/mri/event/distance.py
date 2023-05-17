import numpy as np
import pandas as pd
from analysis.mri.event.base import GAME1EV_base_spct,GAME1EV_base_spat,GAME2EV_base_spct,GAME2EV_base_spat


class GAME1EV_distance_spct(GAME1EV_base_spct):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

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

    def game1ev_distance_spct(self,drop_stalemate=False):
        m1ev = self.genM1ev()
        if drop_stalemate:
            trial_label, accuracy = self.label_trial_drop_stalemate()
        else:
            trial_label, accuracy = self.label_trial_corr()

        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        distance_corr = self.genpm_distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                distance_corr], axis=0)
        return event_data


class GAME1EV_distance_spat(GAME1EV_base_spat):
    """Game1 distance modulation on separate phases for all trials"""
    def __init__(self, behDataPath):
        GAME1EV_base_spat.__init__(self, behDataPath)

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


class GAME1EV_2distance_spct(GAME1EV_base_spct):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

    def genpm_2distance_spct(self, trial_label):
        # generate euclidean distance and manhattan distance
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            eucd = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
            manhd = np.abs(self.behData['ap_diff']) + np.abs(self.behData['dp_diff'])
            # euclean distance event
            eudc_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            eudc_ev['trial_type'] = 'eucd'
            eudc_ev['modulation'] = eucd
            # manhattan distance event
            mand_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            mand_ev['trial_type'] = 'manhd'
            mand_ev['modulation'] = manhd
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            eucd = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
            manhd = np.abs(self.behData['ap_diff']) + np.abs(self.behData['dp_diff'])
            # euclean distance event
            eudc_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            eudc_ev['trial_type'] = 'eucd'
            eudc_ev['modulation'] = eucd
            # manhattan distance event
            mand_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            mand_ev['trial_type'] = 'manhd'
            mand_ev['modulation'] = manhd
        else:
            raise Exception("You need specify behavioral data format.")

        assert len(eudc_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."
        assert len(mand_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        eudc_ev = eudc_ev.iloc[correct_trials_index].copy()
        eudc_ev = eudc_ev.sort_values('onset', ignore_index=True)

        mand_ev = mand_ev.iloc[correct_trials_index].copy()
        mand_ev = mand_ev.sort_values('onset', ignore_index=True)
        return eudc_ev,mand_ev

    def game1ev_2distance_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        eudc_ev,mand_ev = self.genpm_2distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                eudc_ev,mand_ev], axis=0)
        return event_data


class GAME2EV_distance_spct(GAME2EV_base_spct):
    def __init__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)

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


class GAME2EV_distance_center_spct(GAME2EV_base_spct):
    def __init__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)

    def update_behData(self):
        # update the value in pic1_ap,pic1_dp,pic2_ap,pic2_dp(replace all 2.5 and 3.5 to 3)
        for column in ['pic1_ap', 'pic1_dp', 'pic2_ap', 'pic2_dp']:
            self.behData[column] = self.behData[column].replace(2.5, 3)
            self.behData[column] = self.behData[column].replace(3.5, 3)
        # update the ap_diff and dp_diff
        self.behData['ap_diff'] = self.behData['pic2_ap'] - self.behData['pic1_ap']
        self.behData['dp_diff'] = self.behData['pic2_dp'] - self.behData['pic1_dp']
        angle = np.rad2deg(np.arctan2(self.behData['ap_diff'], self.behData['dp_diff']))
        # update the angle
        self.behData['angles'] = angle

    def genpm_distance_center_spct(self, trial_label):
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

    def game2ev_distance_center_spct(self):
        # update the behData
        self.update_behData()
        # generate the event data
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        pmod_distance = self.genpm_distance_center_spct(trial_label)
        # concat the event data
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_distance], axis=0)
        return event_data


class Game2EV_distance_spat(GAME2EV_base_spat):
    """Game2 distance modulation for whole_trials"""

    def __init__(self, behDataPath):
        GAME2EV_base_spat.__init__(self, behDataPath)

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


class GAME2EV_2distance_spct(GAME2EV_base_spct):
    def __init__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)

    def genpm_2distance_spct(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
        else:
            raise Exception("You need specify behavioral data format.")

        eucd = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
        manhd = np.abs(self.behData['ap_diff']) + np.abs(self.behData['dp_diff'])
        # euclean distance event
        eudc_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
        eudc_ev['trial_type'] = 'eucd'
        eudc_ev['modulation'] = eucd
        # manhattan distance event
        mand_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
        mand_ev['trial_type'] = 'manhd'
        mand_ev['modulation'] = manhd

        assert len(eudc_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."
        assert len(mand_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        eudc_ev = eudc_ev.iloc[correct_trials_index].copy()
        eudc_ev = eudc_ev.sort_values('onset', ignore_index=True)

        mand_ev = mand_ev.iloc[correct_trials_index].copy()
        mand_ev = mand_ev.sort_values('onset', ignore_index=True)
        return eudc_ev,mand_ev

    def game2ev_2distance_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        eudc_ev,mand_ev = self.genpm_2distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                eudc_ev,mand_ev], axis=0)
        return event_data