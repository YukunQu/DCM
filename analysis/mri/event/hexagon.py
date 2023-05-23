import numpy as np
import pandas as pd
from analysis.mri.event.base import GAME1EV_base_spct, GAME1EV_base_spat, GAME2EV_base_spct, GAME2EV_base_spat


class EV_hexagon:
    @staticmethod
    def genpm(ev, ifold):
        # ev is reference event of one trial type to provide angle of each trial
        angle = ev['angle']
        pmod_sin = ev.copy()
        pmod_cos = ev.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.round(np.sin(np.deg2rad(ifold * angle)), 2)
        pmod_cos['modulation'] = np.round(np.cos(np.deg2rad(ifold * angle)), 2)
        return pmod_sin, pmod_cos


class GAME1EV_hexagon_spct(GAME1EV_base_spct, EV_hexagon):
    """ game1 event of separate phases with correct trials"""

    def __int__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)
        EV_hexagon.__init__(self)

    def game1ev_hexagon_spct(self, ifold, drop_stalemate=False):
        # base regressors
        m1ev = self.genM1ev()
        # label the correct trials and incorrect trials
        trial_label, accuracy = self.label_trial_corr()

        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # drop stalemate trials
        if drop_stalemate:
            m2ev_corr_dropsm = m2ev_corr.query('stalemate==0')
            deev_corr_dropsm = deev_corr.query('stalemate==0')
            # paramertric modulation regressors
            m2_pmod_sin, m2_pmod_cos = self.genpm(m2ev_corr_dropsm, ifold)
            decision_pmod_sin, decision_pmod_cos = self.genpm(deev_corr_dropsm, ifold)
        else:
            # paramertric modulation regressors
            m2_pmod_sin, m2_pmod_cos = self.genpm(m2ev_corr, ifold)
            decision_pmod_sin, decision_pmod_cos = self.genpm(deev_corr, ifold)
        sin = pd.concat([m2_pmod_sin, decision_pmod_sin], axis=0).sort_values('onset', ignore_index=True)
        cos = pd.concat([m2_pmod_cos, decision_pmod_cos], axis=0).sort_values('onset', ignore_index=True)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                sin, cos], axis=0)
        return event_data


class GAME2EV_hexagon_spct(GAME2EV_base_spct, EV_hexagon):
    """ game2 event of separate phases with correct trials"""

    def __int__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)
        EV_hexagon.__init__(self)

    def game2ev_hexagon_spct(self, ifold):
        # base regressors
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # paramertric modulation regressors
        m2_pmod_sin, m2_pmod_cos = self.genpm(m2ev_corr, ifold)
        decision_pmod_sin, decision_pmod_cos = self.genpm(deev_corr, ifold)
        sin = pd.concat([m2_pmod_sin, decision_pmod_sin], axis=0).sort_values('onset', ignore_index=True)
        cos = pd.concat([m2_pmod_cos, decision_pmod_cos], axis=0).sort_values('onset', ignore_index=True)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                sin, cos], axis=0)
        return event_data


class GAME2EV_hexagon_center_spct(GAME2EV_base_spct, EV_hexagon):
    """ game2 event of separate phases with correct trials"""

    def __int__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)
        EV_hexagon.__init__(self)

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

    def game2ev_hexagon_center_spct(self, ifold):
        # update the behData
        self.update_behData()

        # base regressors
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # paramertric modulation regressors
        m2_pmod_sin, m2_pmod_cos = self.genpm(m2ev_corr, ifold)
        decision_pmod_sin, decision_pmod_cos = self.genpm(deev_corr, ifold)
        sin = pd.concat([m2_pmod_sin, decision_pmod_sin], axis=0).sort_values('onset', ignore_index=True)
        cos = pd.concat([m2_pmod_cos, decision_pmod_cos], axis=0).sort_values('onset', ignore_index=True)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                sin, cos], axis=0)
        return event_data


class GAME1EV_hexagon_spat(GAME1EV_base_spat, EV_hexagon):
    """ game1 event of separate phases with all trials"""

    def __int__(self, behDataPath):
        GAME1EV_base_spat.__init__(self, behDataPath)
        EV_hexagon.__init__(self)

    def game1ev_hexagon_spat(self, ifold):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_sin, pmod_cos = self.genpm(m2ev, ifold)
        # concat all events
        event_data = pd.concat([m1ev, m2ev, deev,
                                pmod_sin, pmod_cos], axis=0)
        return event_data


class Game2EV_hexagon_spat(GAME2EV_base_spat, EV_hexagon):
    """ game2 event of separate phases with all trials"""

    def __int__(self, behDataPath):
        GAME2EV_base_spat.__init__(self, behDataPath)
        EV_hexagon.__init__(self)

    def game2ev_hexagon_spat(self, ifold):
        m1ev = self.genM1ev()
        m2ev = self.genM2ev()
        deev = self.genDeev()
        pmod_sin, pmod_cos = self.genpm(m2ev, ifold)
        event_data = pd.concat([m1ev, m2ev, deev,
                                pmod_sin, pmod_cos], axis=0)
        return event_data
