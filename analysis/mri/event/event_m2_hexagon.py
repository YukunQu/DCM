import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV


class Game1EV_m2hexagon(Game1EV):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self,behDataPath):
        Game1EV.__init__(self,behDataPath)

    def game1ev_m2hexagon(self,ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev_corr,m2ev_error = self.genM2ev(trial_corr)
        deev = self.genDeev_whole_trials()
        response = self.response()
        pmod_sin, pmod_cos = self.genpm(m2ev_corr,ifold)

        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,deev,response,
                                pmod_sin,pmod_cos],axis=0)
        return event_data

    def game1ev_m2hexagon_whole_trials(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        m2ev = self.genM2ev_whole_trials()
        deev = self.genDeev_whole_trials()
        response = self.response()
        pmod_sin, pmod_cos = self.genpm(m2ev,ifold)

        event_data = pd.concat([m1ev,m2ev,deev,response,
                                pmod_sin,pmod_cos],axis=0)
        return event_data


class Game1EV_m2plus_hexagon(Game1EV_m2hexagon):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self,behDataPath):
        Game1EV.__init__(self,behDataPath)

    def genM2ev(self,trial_corr):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = self.behData['cue1.started'] - self.behData['pic2_render.started']
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = self.behData['cue1.started_raw'] - self.behData['pic2_render.started_raw']
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        m2ev_corr = pd.DataFrame(columns=['onset','duration','angle'])
        m2ev_error = pd.DataFrame(columns=['onset','duration','angle'])

        assert len(m2ev) == len(trial_corr), "The number of trial label didn't not same as the number of event-M2."

        for i,trial_label in enumerate(trial_corr):
            if trial_label == True:
                m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
            elif trial_label == False:
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'

        m2ev_corr = m2ev_corr.sort_values('onset', ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset', ignore_index=True)
        return m2ev_corr, m2ev_error

    def game1ev_m2plus_hexagon(self,ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev_corr,m2ev_error = self.genM2ev(trial_corr)
        decision = self.genDeev()
        pmod_sin, pmod_cos = self.genpm(m2ev_corr,ifold)

        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,decision,
                                pmod_sin,pmod_cos],axis=0)
        return event_data


if __name__ == "__main__":
    # set configure
    task = 'game1'
    ifolds = range(6,7)
    runs = range(1,7)
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    data = data.query("(game1_acc>=0.80)and(Age>=18)")
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/m2hexagon_correct_trials/sub-{}/{}fold',
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
                event = Game1EV_m2hexagon(behDataPath)
                event_data = event.game1ev_m2hexagon(ifold)
                tsv_save_path = join(save_dir,template['event_file'].format(subj,task,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)