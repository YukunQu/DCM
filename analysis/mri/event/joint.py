import os
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

    def game1ev_distance_value_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        distance_corr = self.genpm_distance_spct(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                distance_corr,value_corr], axis=0)
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