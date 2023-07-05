import os
import numpy as np
from os.path import join
import pandas as pd
from analysis.mri.event.hexagon import GAME1EV_hexagon_spct,GAME1EV_hexagon_spat
from analysis.mri.event.distance import GAME1EV_distance_spct,GAME1EV_distance_spat,GAME1EV_2distance_spct
from analysis.mri.event.value import GAME1EV_value_spct


class Game1EV_hexagon_distance_spct(GAME1EV_hexagon_spct,GAME1EV_distance_spct):
    def __init__(self,behDataPath):
        GAME1EV_hexagon_spct.__init__(self, behDataPath)
        GAME1EV_distance_spct.__init__(self, behDataPath)

    def game1ev_hexagon_distance_spct(self, ifold):
        # base regressors
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # paramertric modulation regressors
        # hexagonal modulation
        m2_pmod_sin, m2_pmod_cos = self.genpm(m2ev_corr, ifold)
        decision_pmod_sin, decision_pmod_cos = self.genpm(deev_corr, ifold)
        sin = pd.concat([m2_pmod_sin, decision_pmod_sin], axis=0).sort_values('onset', ignore_index=True)
        cos = pd.concat([m2_pmod_cos, decision_pmod_cos], axis=0).sort_values('onset', ignore_index=True)

        # distance modulation
        distance_corr = self.genpm_distance_spct(trial_label)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                sin, cos, distance_corr], axis=0)
        return event_data


class Game1EV_hexagon_distance_spat(GAME1EV_hexagon_spat,GAME1EV_distance_spat):
    def __init__(self, behDataPath):
        GAME1EV_hexagon_spat.__init__(self, behDataPath)
        GAME1EV_distance_spat.__init__(self, behDataPath)

    def game1ev_hexagon_distance_spat(self,ifold):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_distance = self.genpm_distance_spat()
        pmod_sin, pmod_cos = self.genpm(m2ev, ifold)
        event_data = pd.concat([m1ev, m2ev, deev, pmod_sin, pmod_cos, pmod_distance], axis=0)
        return event_data


class GAME1EV_distance_value_spct(GAME1EV_distance_spct,GAME1EV_value_spct):
    def __init__(self, behDataPath):
        GAME1EV_distance_spct.__init__(self, behDataPath)
        GAME1EV_value_spct.__init__(self, behDataPath)

    def genpm_decision_distance_spct(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)

        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff'] ** 2 + self.behData['dp_diff'] ** 2)
        else:
            raise Exception("You need specify behavioral data format.")

        pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
        pmev['trial_type'] = 'decision_corrxdistance'
        pmev['modulation'] = distance
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

    def genpm_M2_value_spct(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
        else:
            raise Exception("You need specify behavioral data format.")

        value1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
        value2ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})

        value1 = []
        value2 = []
        for row in self.behData.itertuples():
            value1.append(np.abs(row.pic1_ap - row.pic2_dp))
            value2.append(np.abs(row.pic2_ap - row.pic1_dp))

        value1ev['trial_type'] = 'value1'
        value1ev['modulation'] = value1

        value2ev['trial_type'] = 'value2'
        value2ev['modulation'] = value2
        assert len(value1ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."
        assert len(value2ev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        value1ev_corr = value1ev.iloc[correct_trials_index].copy()
        value1ev_corr = value1ev_corr.sort_values('onset', ignore_index=True)
        value2ev_corr = value2ev.iloc[correct_trials_index].copy()
        value2ev_corr = value2ev_corr.sort_values('onset', ignore_index=True)
        return value1ev_corr,value2ev_corr

    def game1ev_distance_value_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        m2_distance_corr = self.genpm_distance_spct(trial_label)
        decision_distance_corr = self.genpm_decision_distance_spct(trial_label)
        value1_corr,value2_corr = self.genpm_M2_value_spct(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                m2_distance_corr,decision_distance_corr,
                                value1_corr,value2_corr,value_corr], axis=0)
        return event_data


class GAME1EV_2distance_value_spct(GAME1EV_2distance_spct,GAME1EV_value_spct):
    def __init__(self, behDataPath):
        GAME1EV_2distance_spct.__init__(self, behDataPath)
        GAME1EV_value_spct.__init__(self, behDataPath)

    def game1ev_2distance_value_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        eudc_ev,mand_ev = self.genpm_2distance_spct(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                eudc_ev,mand_ev,value_corr], axis=0)
        return event_data