import os
from os.path import join
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
        trial_label, accuracy = self.label_trial_corr()

        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        if drop_stalemate==True:
            self.label_trials_stalemate()
            self.behData = self.behData.query('stalemate==0') # This operation need be last one.
            trial_label, accuracy = self.label_trial_corr()
        distance_corr = self.genpm_distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                distance_corr], axis=0)
        return event_data


class GAME1EV_hexModdistance_spct(GAME1EV_base_spct):
    # distance effect modulated by hexagonal effect
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

    def genhexsplitM2ev(self, trial_label,phi):
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

        # according to hexagonal effect and split distance into two types(align and misalign)
        corr_trials_angle = m2ev_corr['angle']
        # label alignment trials and misalignment trials according to the angle and Phi
        alignedD_360 = [(a-phi) % 360 for a in corr_trials_angle]
        anglebinNum = [round(a/30)+1 for a in alignedD_360]
        anglebinNum = [1 if a == 13 else a for a in anglebinNum]

        trials_type = []
        for binNum in anglebinNum:
            if binNum in range(1,13,2):
                trials_type.append(f'alignxM2_corr')
            elif binNum in range(2,13,2):
                trials_type.append(f'misalignxM2_corr')
        m2ev_corr['trial_type'] = trials_type
        return m2ev_corr, m2ev_error

    def genpm_hexModdistance_spct(self, trial_label, phi):
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

        # according to hexagonal effect and split distance into two types(align and misalign)
        corr_trials_angle = pmev_corr['angle']
        # label alignment trials and misalignment trials according to the angle and Phi
        alignedD_360 = [(a-phi) % 360 for a in corr_trials_angle]
        anglebinNum = [round(a/30)+1 for a in alignedD_360]
        anglebinNum = [1 if a == 13 else a for a in anglebinNum]

        trials_type = []
        for binNum in anglebinNum:
            if binNum in range(1,13,2):
                trials_type.append(f'alignxdistance')
            elif binNum in range(2,13,2):
                trials_type.append(f'misalignxdistance')
        pmev_corr['trial_type'] = trials_type
        return pmev_corr

    def game1ev_hexModdistance_spct(self,phi,drop_stalemate=False):
        m1ev = self.genM1ev()
        if drop_stalemate:
            trial_label, accuracy = self.label_trials_stalemate()
        else:
            trial_label, accuracy = self.label_trial_corr()

        m2ev_corr, m2ev_error = self.genhexsplitM2ev(trial_label, phi)
        deev_corr, deev_error = self.genDeev(trial_label)
        distance_corr = self.genpm_hexModdistance_spct(trial_label, phi)
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


class GAME1EV_3distance_spct(GAME1EV_base_spct):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

    def genpm_3distance_spct(self, trial_label):
        # generate euclidean distance and manhattan distance
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            eucd = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
            ap = np.abs(self.behData['ap_diff'])
            dp = np.abs(self.behData['dp_diff'])
            # euclean distance event
            eudc_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            eudc_ev['trial_type'] = 'eucd'
            eudc_ev['modulation'] = eucd
            # ap distance event
            ap_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            ap_ev['trial_type'] = 'ap'
            ap_ev['modulation'] = ap
            # dp distance event
            dp_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            dp_ev['trial_type'] = 'dp'
            dp_ev['modulation'] = dp
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            eucd = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
            ap = np.abs(self.behData['ap_diff'])
            dp = np.abs(self.behData['dp_diff'])
            # euclean distance event
            eudc_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            eudc_ev['trial_type'] = 'eucd'
            eudc_ev['modulation'] = eucd
            # ap distance event
            ap_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            ap_ev['trial_type'] = 'ap'
            ap_ev['modulation'] = ap
            # dp distance event
            dp_ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            dp_ev['trial_type'] = 'dp'
            dp_ev['modulation'] = dp
        else:
            raise Exception("You need specify behavioral data format.")

        assert len(eudc_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."
        assert len(ap_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."
        assert len(dp_ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

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

        ap_ev = ap_ev.iloc[correct_trials_index].copy()
        ap_ev = ap_ev.sort_values('onset', ignore_index=True)

        dp_ev = dp_ev.iloc[correct_trials_index].copy()
        dp_ev = dp_ev.sort_values('onset', ignore_index=True)
        return eudc_ev,ap_ev,dp_ev

    def game1ev_3distance_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        eudc_ev,ap_ev,dp_ev = self.genpm_3distance_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                eudc_ev, ap_ev, dp_ev], axis=0)
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


class GAME2EV_hexModdistance_spct(GAME2EV_base_spct):
    # distance effect modulated by hexagonal effect
    def __init__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)

    def genhexsplitM2ev(self, trial_label,phi):
        if self.dformat == 'trial_by_trial':
            onset =  self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
            if 'stalemate' in self.behData.columns:
                m2ev['stalemate'] = self.behData['stalemate']
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
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

        # according to hexagonal effect and split distance into two types(align and misalign)
        corr_trials_angle = m2ev_corr['angle']
        # label alignment trials and misalignment trials according to the angle and Phi
        alignedD_360 = [(a-phi) % 360 for a in corr_trials_angle]
        anglebinNum = [round(a/30)+1 for a in alignedD_360]
        anglebinNum = [1 if a == 13 else a for a in anglebinNum]

        trials_type = []
        for binNum in anglebinNum:
            if binNum in range(1,13,2):
                trials_type.append(f'alignxM2_corr')
            elif binNum in range(2,13,2):
                trials_type.append(f'misalignxM2_corr')
        m2ev_corr['trial_type'] = trials_type
        return m2ev_corr, m2ev_error

    def genpm_hexModdistance_spct(self, trial_label, phi):
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

        # according to hexagonal effect and split distance into two types(align and misalign)
        corr_trials_angle = pmev_corr['angle']
        # label alignment trials and misalignment trials according to the angle and Phi
        alignedD_360 = [(a-phi) % 360 for a in corr_trials_angle]
        anglebinNum = [round(a/30)+1 for a in alignedD_360]
        anglebinNum = [1 if a == 13 else a for a in anglebinNum]

        trials_type = []
        for binNum in anglebinNum:
            if binNum in range(1,13,2):
                trials_type.append(f'alignxdistance')
            elif binNum in range(2,13,2):
                trials_type.append(f'misalignxdistance')
        pmev_corr['trial_type'] = trials_type
        return pmev_corr

    def game2ev_hexModdistance_spct(self,phi):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genhexsplitM2ev(trial_label, phi)
        deev_corr, deev_error = self.genDeev(trial_label)
        distance_corr = self.genpm_hexModdistance_spct(trial_label, phi)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                distance_corr], axis=0)
        return event_data


if __name__ == "__main__":
    ifolds = range(6,7)
    task = 'game2'
    glm_type = 'hexModdistance_spct'
    drop_stalemate = False
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

    phis_file = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/hexagon_spct/estPhi_ROI-EC_circmean_trial-all.csv' # look out
    phis_data = pd.read_csv(phis_file)

    #%%
    for subj in subjects:
        print(f'----sub-{subj}----')

        for ifold in ifolds:
            save_dir = template['save_dir'].format(task, subj, ifold)
            os.makedirs(save_dir, exist_ok=True)

            phi = phis_data.query(f'(sub_id=="sub-{subj}")and(ifold=="{ifold}fold")')['Phi_mean'].values[0]

            for idx in runs:
                run_id = str(idx)
                behav_path = template['behav_path'].format(subj, subj, task, run_id)
                event = GAME2EV_hexModdistance_spct(behav_path)
                event_data = event.game2ev_hexModdistance_spct(phi)
                tsv_save_path = join(save_dir, template['event_file'].format(subj, task, run_id))
                event_data.to_csv(tsv_save_path, sep='\t', index=False)