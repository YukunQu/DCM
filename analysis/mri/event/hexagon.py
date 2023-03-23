import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.base import Game1EV_base_spct,Game2EV_base_spct


class Game1EV_hexagon_spct(Game1EV_base_spct):
    """ event separate phases correct trials"""
    def __int__(self, behDataPath):
        Game1EV_base_spct.__init__(self, behDataPath)

    @staticmethod
    def genpm(ev, ifold):
        # ev is reference event of one trial type to provide angle of each trial
        angle = ev['angle']
        pmod_sin = ev.copy()
        pmod_cos = ev.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.round(np.sin(np.deg2rad(ifold * angle)),2)
        pmod_cos['modulation'] = np.round(np.cos(np.deg2rad(ifold * angle)),2)
        return pmod_sin, pmod_cos

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


class Game2EV_hexagon_spct(Game2EV_base_spct):
    def __int__(self, behDataPath):
        Game2EV_base_spct.__init__(self, behDataPath)

    @staticmethod
    def genpm(ev, ifold):
        angle = ev['angle']
        pmod_sin = ev.copy()
        pmod_cos = ev.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold * angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold * angle))
        return pmod_sin, pmod_cos

    def game2ev_hexagon_spct(self, ifold):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        pmod_sin, pmod_cos = self.genpm(m2ev_corr, ifold)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_sin, pmod_cos], axis=0)
        return event_data


class Game1EV_hexagon_spat(Game1EV_base_spat):
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