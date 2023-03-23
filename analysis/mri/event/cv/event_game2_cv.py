import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.hexagon.event_hexagon_spct import Game2EV_hexagon_spct


class Game2_cv_hexagon_spct(Game2EV_hexagon_spct):
    def __init__(self, behDataPath):
        Game2EV_hexagon_spct.__init__(self, behDataPath)

    def genpm_alignPhi(self, ev, ifold, phi):
        # generate parametric modulation for test GLM
        angle = ev['angle']
        pmod_alignPhi = ev.copy()
        pmod_alignPhi['modulation'] = np.round(np.cos(np.deg2rad(ifold * (angle - phi))), 2)
        pmod_alignPhi['trial_type'] = 'alignPhi'
        return pmod_alignPhi


    def game2ev_cv_hexagon_spct(self, ifold, phi):
        # base regressors
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # paramertric modulation regressors
        m2_alignPhi = self.genpm_alignPhi(m2ev_corr, ifold, phi)
        decision_alignPhi = self.genpm_alignPhi(deev_corr,ifold, phi)
        alignPhi = pd.concat([m2_alignPhi,decision_alignPhi],axis=0).sort_values('onset', ignore_index=True)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                alignPhi], axis=0)
        return event_data


class Game2_cv_hexagon_distance_spct(Game2_cv_hexagon_spct):
    def __init__(self, behDataPath):
        Game2_cv_hexagon_spct.__init__(self, behDataPath)

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

    def game2ev_hexagon_distance_spct(self, ifold, phi):
        # base regressors
        m1ev = self.genM1ev()
        trial_label, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_label)
        deev_corr, deev_error = self.genDeev(trial_label)

        # paramertric modulation regressors
        # hexagonal modulation
        m2_alignPhi = self.genpm_alignPhi(m2ev_corr, ifold, phi)
        decision_alignPhi = self.genpm_alignPhi(deev_corr,ifold, phi)
        alignPhi = pd.concat([m2_alignPhi,decision_alignPhi],axis=0).sort_values('onset', ignore_index=True)

        # distance modulation
        m2ev_distance = self.genpm_distance_spct(trial_label)
        m2ev_distance['trial_type'] = 'M2_corrx' + m2ev_distance['trial_type']
        distance_pm = m2ev_distance['modulation']
        deev_distance = deev_corr.copy()
        deev_distance['modulation'] = distance_pm
        deev_distance['trial_type'] = deev_distance['trial_type'] + 'xdistance'

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                alignPhi,m2ev_distance,deev_distance], axis=0)
        return event_data


def gen_sub_event(subjects):
    """
    generate game2 evnet based on game1's Phi
    """
    task = 'game2'
    # set folds and runs for cross validation
    runs = range(1, 3)
    ifolds= range(6, 7)
    parameters = {'behav_path': r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/'
                                r'fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                  'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'
                              r'cv_hexagon_distance_spct/sub-{}/{}fold',      # look out
                  'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}

    # set Phi estimated from specific ROI
    glm_type = 'hexagon_distance_spct'
    phis_file = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/hexagon_distance_spct/estPhi_ROI-EC_circmean_trial-all.csv'  # look out
    phis_data = pd.read_csv(phis_file)

    for ifold in ifolds:
        for subj in subjects:
            subj = str(subj).zfill(3)
            print(ifold,'fold:sub-',subj, "started.")
            phi = phis_data.query(f'(sub_id=="sub-{subj}")and(ifold=="{ifold}fold")')['Phi_mean'].values[0]
            for run_id in runs:
                behDataPath = parameters['behav_path'].format(subj, subj, task, run_id)
                if glm_type == 'hexagon_spct':
                    event = Game2_cv_hexagon_spct(behDataPath)
                    event_data = event.game2ev_cv_hexagon_spct(ifold,phi)  # lookout
                elif glm_type == 'hexagon_distance_spct':
                    event = Game2_cv_hexagon_distance_spct(behDataPath)
                    event_data = event.game2ev_hexagon_distance_spct(ifold,phi) # lookout
                # save
                save_dir = parameters['save_dir'].format(task,subj, ifold)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                tsv_save_path = join(save_dir, parameters['event_file'].format(subj, task, run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'game2_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects_list = [p.split('-')[-1] for p in pid]
    gen_sub_event(subjects_list)