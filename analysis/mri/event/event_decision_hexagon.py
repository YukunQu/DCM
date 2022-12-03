import os
from os.path import join
import numpy as np
import pandas as pd
from analysis.mri.event.game1_event import Game1EV

class Game1EV_decision_hexagon(Game1EV):
    # A variant of event generator only care about the hexagonal effect on M2.
    def __init__(self,behDataPath):
        Game1EV.__init__(self,behDataPath)

    def genM2ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic2_render.started'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic2_render.started_raw'] - self.starttime
            duration = [2.5] * len(self.behData)
            angle = self.behData['angles']
            m2ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m2ev['trial_type'] = 'M2'
            m2ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m2ev = m2ev.sort_values('onset', ignore_index=True)
        return m2ev

    def game1ev_decision_hexagon(self,ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev = self.genM2ev()
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_sin, pmod_cos = self.genpm(deev_corr,ifold)

        event_data = pd.concat([m1ev,m2ev,deev_corr,deev_error,
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
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/decision_hexagon_correct_trials/sub-{}/{}fold',
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
                event = Game1EV_decision_hexagon(behDataPath)
                event_data = event.game1ev_decision_hexagon(ifold)
                tsv_save_path = join(save_dir,template['event_file'].format(subj,task,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)