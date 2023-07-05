import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.base import GAME1EV_base_spct,GAME2EV_base_spct



class GAME1EV_value_spct(GAME1EV_base_spct):
    # A variant of event generator for GAME1's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

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


class GAME1EV_hexModvalue_spct(GAME1EV_base_spct):
    # A variant of event generator for GAME1's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

    def genhexsplitDeev(self, trial_label,phi):
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

        # according to hexagonal effect and split distance into two types(align and misalign)
        corr_trials_angle = deev_corr['angle']
        # label alignment trials and misalignment trials according to the angle and Phi
        alignedD_360 = [(a-phi) % 360 for a in corr_trials_angle]
        anglebinNum = [round(a/30)+1 for a in alignedD_360]
        anglebinNum = [1 if a == 13 else a for a in anglebinNum]

        trials_type = []
        for binNum in anglebinNum:
            if binNum in range(1,13,2):
                trials_type.append(f'alignxDecision_corr')
            elif binNum in range(2,13,2):
                trials_type.append(f'misalignxDecision_corr')
        deev_corr['trial_type'] = trials_type
        return deev_corr, deev_error

    def genpm_hexModvalue_spct(self, trial_label, phi):
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

        # according to hexagonal effect and split value into two types(align and misalign)
        corr_trials_angle = pmev_corr['angle']
        # label alignment trials and misalignment trials according to the angle and Phi
        alignedD_360 = [(a-phi) % 360 for a in corr_trials_angle]
        anglebinNum = [round(a/30)+1 for a in alignedD_360]
        anglebinNum = [1 if a == 13 else a for a in anglebinNum]

        trials_type = []
        for binNum in anglebinNum:
            if binNum in range(1,13,2):
                trials_type.append(f'alignxvalue')
            elif binNum in range(2,13,2):
                trials_type.append(f'misalignxvalue')
        pmev_corr['trial_type'] = trials_type
        return pmev_corr

    def game1ev_hexModvalue_spct(self,phi):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genhexsplitDeev(trial_label,phi)
        value_corr = self.genpm_hexModvalue_spct(trial_label,phi)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                value_corr], axis=0)
        return event_data


class GAME1EV_pure_value_spct(GAME1EV_base_spct):
    # A variant of event generator for GAME1's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

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
                value.append(row.pic1_ap - row.pic2_dp)
            elif rule == '1D2A':
                value.append(row.pic2_ap - row.pic1_dp)

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

    def game1ev_pure_value_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                value_corr], axis=0)
        return event_data


class GAME1EV_ap_spct(GAME1EV_base_spct):
    # A variant of event generator for GAME1's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

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
                value.append(row.pic1_ap)
            elif rule == '1D2A':
                value.append(row.pic2_ap)

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

    def game1ev_ap_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                value_corr], axis=0)
        return event_data


class GAME1EV_dp_spct(GAME1EV_base_spct):
    # A variant of event generator for GAME1's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

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
                value.append(row.pic2_dp)
            elif rule == '1D2A':
                value.append(row.pic1_dp)

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

    def game1ev_dp_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                value_corr], axis=0)
        return event_data


class GAME1EV_apdp_spct(GAME1EV_base_spct):
    # A variant of event generator for GAME1's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME1EV_base_spct.__init__(self, behDataPath)

    def genpm_apdp_spct(self, trial_label):
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

        apev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
        dpev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})

        ap,dp = [],[]
        for row in self.behData.itertuples():
            rule = row.fightRule
            if rule == '1A2D':
                ap.append(row.pic1_ap)
                dp.append(row.pic1_dp)
            elif rule == '1D2A':
                ap.append(row.pic2_ap)
                dp.append(row.pic2_dp)

        apev['trial_type'] = 'ap'
        apev['modulation'] = ap
        dpev['trial_type'] = 'dp'
        dpev['modulation'] = dp
        assert len(dpev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."
        assert len(apev) == len(trial_label), "The number of trial label didn't not same as the number of event-M2."

        correct_trials_index = []
        error_trials_index = []
        for i, label in enumerate(trial_label):
            if label:
                correct_trials_index.append(i)
            elif not label:
                error_trials_index.append(i)
            else:
                raise ValueError("The trial label should be True or False.")

        apev_corr = apev.iloc[correct_trials_index].copy()
        apev_corr = apev_corr.sort_values('onset', ignore_index=True)

        dpev_corr = dpev.iloc[correct_trials_index].copy()
        dpev_corr = dpev_corr.sort_values('onset', ignore_index=True)
        return apev_corr,dpev_corr

    def game1ev_apdp_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        apev_corr,dpev_corr = self.genpm_apdp_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                apev_corr,dpev_corr], axis=0)
        return event_data



class GAME2EV_value_spct(GAME2EV_base_spct):
    # A variant of event generator for GAME2's value parametric modulation at decision.
    def __init__(self, behDataPath):
        GAME2EV_base_spct.__init__(self, behDataPath)

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

    def game2ev_value_spct(self):
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)
        value_corr = self.genpm_value_spct(trial_label)
        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                value_corr], axis=0)
        return event_data


if __name__ == "__main__":
    ifolds = range(6,7)
    task = 'game1'
    glm_type = 'hexModvalue_spct'
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
                event = GAME1EV_hexModvalue_spct(behav_path)
                event_data = event.game1ev_hexModvalue_spct(phi)
                tsv_save_path = join(save_dir, template['event_file'].format(subj, task, run_id))
                event_data.to_csv(tsv_save_path, sep='\t', index=False)