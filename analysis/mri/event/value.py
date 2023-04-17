import numpy as np
import pandas as pd
from analysis.mri.event.base import Game1EV_base_spct


class GAME1EV_value_spct(Game1EV_base_spct):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        Game1EV_base_spct.__init__(self, behDataPath)

    def genpm_value_spct(self, trial_label):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
        else:
            raise Exception("You need specify behavioral data format.")

        pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})

        value = []
        for row in self.behData.itertuples():
            rule = row.fightRule
            if rule == '1A2D':
                value.append(np.abs(row.pic1_ap - row.pic2_dp))
            elif rule == '1D2A':
                value.append(np.abs(row.pic2_ap - row.pic1_dp))

        pmev['trial_type'] = 'value'
        pmev['modulation'] = value
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

    def game1ev_value_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                value_corr], axis=0)
        return event_data