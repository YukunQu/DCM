import os
from os.path import join
import numpy as np
import pandas as pd


class Game1EV(object):
    """"""
    def __init__(self,behDataPath):
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.dformat = None

    def game1_dformat(self):
        columns = self.behData.columns
        if 'fix_start_cue.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            raise Exception("You need specify behavioral data format.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fix_start_cue.started'].min()
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min() - 1
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime

    def label_trial_corr(self):
        self.behData = self.behData.fillna('None')
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['resp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['resp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            raise Exception("You need specify behavioral data format.")

        trial_corr = []
        for keyResp,row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                else:
                    correctAns = 2
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                else:
                    correctAns = 1
            else:
                raise Exception("None of rule have been found in the file.")
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData),3)
        return trial_corr,accuracy

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['pic1_render.started'] - self.starttime
            duration = self.behData['pic2_render.started'] - self.behData['pic1_render.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['pic1_render.started_raw'] - self.starttime
            duration = self.behData['pic2_render.started_raw'] - self.behData['pic1_render.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset',ignore_index=True)
        return m1ev

    def genM2ev(self,trial_corr):
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

        m2ev_corr = m2ev_corr.sort_values('onset',ignore_index=True)
        m2ev_error = m2ev_error.sort_values('onset',ignore_index=True)
        return m2ev_corr, m2ev_error

    def genDeev(self,trial_corr):
        # generate the event of decision
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = self.behData['cue1_2.started'] - self.behData['cue1.started']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = self.behData['cue1_2.started_raw'] - self.behData['cue1.started_raw']
            angle = self.behData['angles']
            deev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            deev['trial_type'] = 'decision'
            deev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")

        deev_corr = pd.DataFrame(columns=['onset','duration','angle'])
        deev_error = pd.DataFrame(columns=['onset','duration','angle'])

        assert len(deev) == len(trial_corr), "The number of trial label didn't not  same as the number of event-decision."

        for i,trial_label in enumerate(trial_corr):
            if trial_label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif trial_label == False:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'

        deev_corr = deev_corr.sort_values('onset',ignore_index=True)
        deev_error = deev_error.sort_values('onset',ignore_index=True)
        return deev_corr, deev_error

    def pressButton(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        pbev = pbev.sort_values('onset',ignore_index=True)
        return pbev

    def genpm(self,m2ev_corr,ifold):
        angle = m2ev_corr['angle']
        pmod_sin = m2ev_corr.copy()
        pmod_cos = m2ev_corr.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold*angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold*angle))
        return pmod_sin, pmod_cos

    def game1ev(self,ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        pbev = self.pressButton()
        trial_corr,accuracy = self.label_trial_corr()
        m2ev_corr,m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pmod_sin, pmod_cos = self.genpm(m2ev_corr,ifold)

        event_data = pd.concat([m1ev,m2ev_corr,m2ev_error,deev_corr,deev_error,pbev,
                                pmod_sin,pmod_cos],axis=0)
        return event_data


class Game2EV(object):
    """"""
    def __init__(self, behDataPath):
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.behData = self.behData.fillna('None')
        self.dformat = None

    def game1_dformat(self):
        columns = self.behData.columns
        if 'fixation.started' in columns:
            self.dformat = 'trial_by_trial'
        elif 'fixation.started_raw' in columns:
            self.dformat = 'summary'
        else:
            print("The data is not game2 behavioral data.")

    def cal_start_time(self):
        self.game1_dformat()
        if self.dformat == 'trial_by_trial':
            starttime = self.behData['fixation.started'].min()
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min()
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime

    def genM1ev(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['testPic1.started'] - self.starttime
            duration = self.behData['testPic2.started'] - self.behData['testPic1.started']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['testPic1.started_raw'] - self.starttime
            duration = self.behData['testPic2.started_raw'] - self.behData['testPic1.started_raw']
            angle = self.behData['angles']

            m1ev = pd.DataFrame({'onset': onset, 'duration': duration, 'angle': angle})
            m1ev['trial_type'] = 'M1'
            m1ev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        m1ev = m1ev.sort_values('onset',ignore_index=True)
        return m1ev

    def label_trial_corr(self):
        if self.dformat == 'trial_by_trial':
            keyResp_list = self.behData['dResp.keys']
        elif self.dformat == 'summary':
            keyResp_tmp = self.behData['dResp.keys_raw']
            keyResp_list = []
            for k in keyResp_tmp:
                if k == 'None':
                    keyResp_list.append(k)
                else:
                    keyResp_list.append(k[1])
        else:
            print("You need specify behavioral data format.")

        trial_corr = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                else:
                    correctAns = 2
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                else:
                    correctAns = 1
            if (keyResp == 'None') or (keyResp == None):
                trial_corr.append(False)
            elif int(keyResp) == correctAns:
                trial_corr.append(True)
            else:
                trial_corr.append(False)
        accuracy = np.round(np.sum(trial_corr) / len(self.behData), 3)
        return trial_corr, accuracy

    def genM2ev(self, trial_corr):

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

        assert len(m2ev) == len(trial_corr), "The number of trial label didn't not  same as the number of " \
                                             "event-decision. "

        m2ev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        m2ev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                m2ev_corr = m2ev_corr.append(m2ev.iloc[i])
            elif trial_label == False:
                m2ev_error = m2ev_error.append(m2ev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        m2ev_corr['trial_type'] = 'M2_corr'
        m2ev_error['trial_type'] = 'M2_error'
        return m2ev_corr, m2ev_error

    def genDeev(self,trial_corr):
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

        assert len(deev) == len(trial_corr), "The number of trial label didn't not  same as the number of " \
                                             "event-decision. "

        deev_corr = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        deev_error = pd.DataFrame(columns=['onset', 'duration', 'angle'])
        for i, trial_label in enumerate(trial_corr):
            if trial_label == True:
                deev_corr = deev_corr.append(deev.iloc[i])
            elif trial_label == False:
                deev_error = deev_error.append(deev.iloc[i])
            else:
                raise ValueError("The trial label should be True or False.")
        deev_corr['trial_type'] = 'decision_corr'
        deev_error['trial_type'] = 'decision_error'
        return deev_corr, deev_error

    def pressButton(self):
        if self.dformat == 'trial_by_trial':
            onset = self.behData['cue1.started'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        elif self.dformat == 'summary':
            onset = self.behData['cue1.started_raw'] - self.starttime
            duration = 0
            angle = self.behData['angles']
            pbev = pd.DataFrame({'onset':onset,'duration':duration,'angle':angle})
            pbev['trial_type'] = 'pressButton'
            pbev['modulation'] = 1
        else:
            raise Exception("You need specify behavioral data format.")
        return pbev

    def genpm(self, m2ev_corr, ifold):
        angle = m2ev_corr['angle']
        pmod_sin = m2ev_corr.copy()
        pmod_cos = m2ev_corr.copy()
        pmod_sin['trial_type'] = 'sin'
        pmod_cos['trial_type'] = 'cos'
        pmod_sin['modulation'] = np.sin(np.deg2rad(ifold * angle))
        pmod_cos['modulation'] = np.cos(np.deg2rad(ifold * angle))

        return pmod_sin, pmod_cos

    def game2ev(self, ifold):
        self.starttime = self.cal_start_time()
        m1ev = self.genM1ev()
        trial_corr, accuracy = self.label_trial_corr()
        m2ev_corr, m2ev_error = self.genM2ev(trial_corr)
        deev_corr, deev_error = self.genDeev(trial_corr)
        pbev = self.pressButton()
        pmod_sin, pmod_cos = self.genpm(m2ev_corr, ifold)

        event_data = pd.concat([m1ev, m2ev_corr, m2ev_error,deev_corr, deev_error,pbev,
                                pmod_sin, pmod_cos], axis=0)
        return event_data


def gen_sub_event(task, subjects):
    if task == 'game1':
        runs = range(1,7)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/separate_hexagon_old/sub-{}/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    elif task == 'game2':
        runs = range(1,3)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/separate_hexagon_old/sub-{}/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")

    ifolds = range(4,9)

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
                if task == 'game1':
                    event = Game1EV(behDataPath)
                    event_data = event.game1ev(ifold)
                elif task == 'game2':
                    event = Game2EV(behDataPath)
                    event_data = event.game2ev(ifold)
                else:
                    raise Exception("The type of task is wrong.")
                tsv_save_path = join(save_dir,template['event_file'].format(subj,task,run_id))
                event_data.to_csv(tsv_save_path, sep="\t", index=False)


if __name__ == "__main__":
    task = 'game1'
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv,sep='\t')
    data = participants_data.query(f'{task}_fmri==1')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]
    gen_sub_event(task, subjects)