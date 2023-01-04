import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV
from analysis.mri.event.game2_event import Game2EV

class Game1EV_value(Game1EV):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)

    def genpm_value(self,trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            value = 1   # ************************
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'value'
            pmev['modulation'] = value
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)  # ************************
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")

        pmev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        assert len(pmev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."

        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                pmev_corr = pmev_corr.append(pmev.iloc[i])
            elif trial_label == False:
                continue
            elif trial_label == None:
                continue
            else:
                raise ValueError("The trial label should be True,False or None.")
        pmev_corr = pmev_corr.sort_values('onset', ignore_index=True)
        return pmev_corr

    def game1ev_distance(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_distance = self.genpm_value(trial_corr)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_distance], axis=0)
        return event_data


class Game1EV_distance_whole_trials(Game1EV):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self, behDataPath):
        Game1EV.__init__(self, behDataPath)

    def genpm_distance_whole_trials(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")
        pmev = pmev.sort_values('onset', ignore_index=True)
        return pmev

    def game1ev_distance_whole_trials(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        m2ev = self.genM2ev_whole_trials()
        deev = self.genDeev_whole_trials()
        pmod_distance = self.genpm_distance_whole_trials()
        event_data = pd.concat([m1ev, m2ev, deev,
                                pmod_distance], axis=0)
        return event_data


class Game2EV_distance(Game2EV):
    def __init__(self, behDataPath):
        Game2EV.__init__(self, behDataPath)

    def genpm_distance(self,trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")

        pmev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        assert len(pmev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."
        for i, label in enumerate(trial_corr):
            if label == True:
                pmev_corr = pmev_corr.append(pmev.iloc[i])
            elif (label == False)or(label==None):
                continue
            else:
                raise ValueError("The trial label should be True,False or None.")
        pmev_corr = pmev_corr.sort_values('onset', ignore_index=True)
        return pmev_corr

    def game2ev_distance(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_distance = self.genpm_distance(trial_corr)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error, deev_corr, deev_error,
                                pmod_distance], axis=0)
        return event_data



class Game2EV_distance_whole_trials(Game2EV):
    def __init__(self, behDataPath):
        Game2EV.__init__(self, behDataPath)

    def genpm_distance_whole_trials(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic2.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)

            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        elif self.dformat == 'summary':
            onset = self.behData['testPic2.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            distance = np.sqrt(self.behData['ap_diff']**2 + self.behData['dp_diff']**2)
            pmev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            pmev['trial_type'] = 'distance'
            pmev['modulation'] = distance
        else:
            raise Exception("You need specify behavioral data format.")
        return pmev

    def game2ev_distance_whole_trials(self):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        m2ev = self.genM2ev_whole_trials()
        deev = self.genDeev_whole_trials()
        pmod_distance = self.genpm_distance_whole_trials()

        event_data = pd.concat([m1ev, m2ev, deev, pmod_distance], axis=0)
        return event_data


if __name__ == "__main__":
    # set configure
    task = 'game2'
    ifolds = range(6,7)
    runs = range(1,3)
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query(f'{task}_fmri>0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/distance/sub-{}/{}fold',
                'event_file':'sub-{}_task-{}_run-{}_events.tsv'}

    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))

        for ifold in ifolds:
            save_dir = template['save_dir'].format(task,subj,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in runs:
                run_id = str(idx)
                behDataPath = template['behav_path'].format(subj,subj,task,run_id)
                event = Game2EV_distance(behDataPath)
                event_data = event.game2ev_distance()
                tsv_save_path = join(save_dir,template['event_file'].format(subj,task,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)