import os
from os.path import join
import numpy as np
import pandas as pd


def label_trial_img(df,map,pic_id):
    pic_ap = df[f'{pic_id}_ap']
    pic_dp = df[f'{pic_id}_dp']
    img_label = map.query(f"(Attack_Power=={pic_ap})and(Defence_Power=={pic_dp})")['Image_label'].values[0]
    return 'pos'+str(img_label)


class Game1RSAEV(object):
    """"""
    def __init__(self,behDataPath):
        self.behDataPath = behDataPath
        self.behData = pd.read_csv(behDataPath)
        self.behData = self.behData.dropna(axis=0, subset=['pairs_id'])
        self.behData = self.behData.fillna('None')
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
            starttime = self.behData['fix_start_cue.started'][1]
        elif self.dformat == 'summary':
            starttime = self.behData['fixation.started_raw'].min() - 1
        else:
            raise Exception("You need specify behavioral data format.")
        return starttime


    def labelImage(self):
        x = range(1,6)
        y = range(1,6)
        d1,d2 = np.meshgrid(x,y)
        d1 = d1.reshape(-1)
        d2 = d2.reshape(-1)
        map = pd.DataFrame({'Image_label':range(1,len(d1)+1),'Attack_Power':d1,'Defence_Power':d2})
        self.behData['pic1_label'] = self.behData.apply(label_trial_img,axis=1,args=(map,'pic1'))
        self.behData['pic2_label'] = self.behData.apply(label_trial_img,axis=1,args=(map,'pic2'))


    def rsa_ev(self):
        self.starttime = self.cal_start_time()
        ev = pd.DataFrame(columns=['onset','duration','trial_type'])
        if self.dformat == 'trial_by_trial':
            self.behData = self.behData.sort_values('pic1_render.started', ascending=True)
            self.labelImage()
            for index,row in self.behData.iterrows():
                pic1_onset = row['pic1_render.started'] - self.starttime
                pic1_duration = row['pic2_render.started'] - row['pic1_render.started']
                label = row['pic1_label']
                ev = ev.append({'onset':pic1_onset,'duration':pic1_duration,'trial_type':label,'modulation':1},
                               ignore_index=True)

                pic2_onset = row['pic2_render.started'] - self.starttime
                pic2_duration = 2.5
                label = row['pic2_label']
                ev = ev.append({'onset':pic2_onset,'duration':pic2_duration,'trial_type':label,'modulation':1},
                               ignore_index=True)
        elif self.dformat == 'summary':
            self.behData = self.behData.sort_values('pic1_render.started_raw', ascending=True)
            self.labelImage()
            for index,row in self.behData.iterrows():
                pic1_onset = row['pic1_render.started_raw'] - self.starttime
                pic1_duration = row['pic2_render.started_raw'] - row['pic1_render.started_raw']
                label = row['pic1_label']
                ev = ev.append({'onset':pic1_onset,'duration':pic1_duration,'trial_type':label,'modulation':1},
                               ignore_index=True)

                pic2_onset = row['pic2_render.started_raw'] - self.starttime
                pic2_duration = 2.5
                label = row['pic2_label']
                ev = ev.append({'onset':pic2_onset,'duration':pic2_duration,'trial_type':label,'modulation':1},
                               ignore_index=True)
        else:
            raise Exception("You need specify behavioral data format.")
        return ev

def gen_sub_rsa_event(task, subjects):
    if task == 'game1':
        runs = range(1,7)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game1/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/{}/rsa/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    elif task == 'game2':
        runs = range(1,3)
        template = {'behav_path':r'/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/fmri_task-game2-test/sub-{}_task-{}_run-{}.csv',
                    'save_dir':r'/mnt/workdir/DCM/BIDS/derivatives/Events/sub-{}/{}/rsa/{}fold',
                    'event_file':'sub-{}_task-{}_run-{}_events.tsv'}
    else:
        raise Exception("The type of task is wrong.")

    ifolds = range(6,7)

    for subj in subjects:
        subj = str(subj).zfill(3)
        print('----sub-{}----'.format(subj))

        for ifold in ifolds:
            save_dir = template['save_dir'].format(subj,task,ifold)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in runs:
                run_id = str(idx)
                behDataPath = template['behav_path'].format(subj,subj,task,run_id)
                if task == 'game1':
                    event = Game1RSAEV(behDataPath)
                    event_data = event.rsa_ev()
                elif task == 'game2':
                    pass
                    #event = Game2EV(behDataPath)
                    #event_data = event.game2ev(ifold)
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
    subjects = [p.split('_')[-1] for p in pid]

    gen_sub_rsa_event(task,subjects)