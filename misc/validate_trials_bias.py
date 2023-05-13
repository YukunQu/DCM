import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('seaborn-pastel')
sns.set_style('white')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})
sns.set_theme(style="white")



def angle2binNum(angles):
    alignedD_360 = [a % 360 for a in angles]
    anglebinNum = [round(a/30)+1 for a in alignedD_360]
    anglebinNum = [1 if binN == 13 else binN for binN in anglebinNum]

    # Compute pie slices
    N = int(360/30)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    binsAngleNum = []
    for binNum in range(1,13):
        binAngleNum = 0
        for a in anglebinNum:
            if a == binNum:
                binAngleNum +=1
        binsAngleNum.append(binAngleNum)
    return binsAngleNum


def plotAngleRadar(angles):
    binsAngleNum = angle2binNum(angles)
    N = int(360/30)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = 2*np.pi / (N+1)
    ax = plt.subplot(projection='polar')
    ax.bar(theta, binsAngleNum,width=width,bottom=0.0, alpha=0.5)
    ax.set_yticks([0,500,1000,1500])
    plt.show()


def plot_angle_acc(corr_angles,error_angles):
    import math
    binsAngle_corrNum = np.array(angle2binNum(corr_angles))
    binsAngle_errorNum = np.array(angle2binNum(error_angles))
    binsAngle_acc = binsAngle_corrNum/(binsAngle_corrNum+binsAngle_errorNum)

    N = int(360/30)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = 2*np.pi / (N+1)
    ax = plt.subplot(projection='polar')
    binsAngle_acc = binsAngle_acc - 0.5
    ax.bar(theta, binsAngle_acc, width=width, bottom=0.0, alpha=0.5)
    ax.set_yticks([0,0.2,0.4])
    ax.set_yticklabels([0.5,0.7,0.9])
    plt.show()

#%%
# example for trial sample
pid = ['sub-180']
run_template = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/hexagon_spat/{}/6fold/{}_task-game1_run-{}_events.tsv'
trials_angle = []
for sub_id in pid:
    for run_id in range(1,7):
        run_path = run_template.format(sub_id,sub_id,run_id)
        run_file = pd.read_csv(run_path,sep='\t')
        run_angles = run_file.query('trial_type=="M2"')['angle'].to_list()
        trials_angle.extend(run_angles)
plotAngleRadar(trials_angle)

#%%
"""
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()
pid = ['sub-196','sub-196']

run_template = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/cv_train1/{}/6fold/{}_task-game1_run-{}_events.tsv'
odd_trials_angle = []
even_trials_angle = []
for sub_id in pid:
    for run_id in range(1,7):
        run_path = run_template.format(sub_id,sub_id,run_id)
        run_file = pd.read_csv(run_path,sep='\t')
        run_odd = run_file.query('trial_type=="M2_corr_odd"')['angle'].to_list()
        run_even = run_file.query('trial_type=="M2_corr_even"')['angle'].to_list()
        odd_trials_angle.extend(run_odd)
        even_trials_angle.extend(run_even)

#%%
plotAngleRadar(odd_trials_angle)
plotAngleRadar(even_trials_angle)
"""
#%%
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
pid = data['Participant_ID'].to_list()

#  calculate the angle
run_template = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/distance_spct/{}/6fold/{}_task-game1_run-{}_events.tsv'
corr_trials_angle = []
error_trials_angle = []
acc_trials_angle = []
for sub_id in pid:
    for run_id in range(1,7):
        run_path = run_template.format(sub_id,sub_id,run_id)
        run_file = pd.read_csv(run_path,sep='\t')
        run_corr = run_file.query('trial_type=="M2_corr"')['angle'].to_list()
        run_error = run_file.query('trial_type=="M2_error"')['angle'].to_list()
        corr_trials_angle.extend(run_corr)
        error_trials_angle.extend(run_error)

plotAngleRadar(corr_trials_angle)
plotAngleRadar(error_trials_angle)
plot_angle_acc(corr_trials_angle,error_trials_angle)


#%%
import os
import numpy as np
import pandas as pd
from os.path import join
from analysis.mri.event.base import GAME1EV
# get the angle distribution of dogfall trials

class GAME1EV_tmp(GAME1EV):
    def __init__(self, behDataPath):
        GAME1EV.__init__(self, behDataPath)

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

        angle = self.behData['angles']
        trial_corr = []
        fr = []
        fresult = []
        for keyResp, row in zip(keyResp_list, self.behData.itertuples()):
            rule = row.fightRule
            fr.append(rule)
            if rule == '1A2D':
                fight_result = row.pic1_ap - row.pic2_dp
                if fight_result > 0:
                    correctAns = 1
                elif fight_result <0:
                    correctAns = 2
                elif fight_result == 0:
                    correctAns = -1
                else:
                    raise Exception("fight result is not a number.")
            elif rule == '1D2A':
                fight_result = row.pic2_ap - row.pic1_dp
                if fight_result > 0:
                    correctAns = 2
                elif fight_result <0:
                    correctAns = 1
                elif fight_result == 0:
                    correctAns = -1
                else:
                    raise Exception("fight result is not a number.")
            else:
                raise Exception("None of rule have been found in the file.")
            fresult.append(fight_result)
        pic1_ap = self.behData['pic1_ap']
        pic2_ap = self.behData['pic2_ap']
        pic1_dp = self.behData['pic1_dp']
        pic2_dp = self.behData['pic2_dp']
        return fr,fresult,angle,pic1_ap,pic2_ap,pic1_dp,pic2_dp



ifolds = [6]
task = 'game1'
glm_type = 'hexagon_spct'
template = {'save_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events/{}/'+glm_type+'/sub-{}/{}fold',
            'event_file': 'sub-{}_task-{}_run-{}_events.tsv'}

if task == 'game1':
    runs = range(1, 7)
    template['behav_path'] = '/mnt/workdir/DCM/sourcedata/sub_{}/Behaviour/' \
                             'fmri_task-game1/sub-{}_task-{}_run-{}.csv'

participants_data = pd.read_csv('/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
subjects = participants_data.query(f'{task}_fmri>=0.5')['Participant_ID'].str.split('-').str[-1].str.zfill(3)

df = pd.DataFrame()
for subj in subjects:
    print(f'----sub-{subj}----')

    for ifold in ifolds:
        save_dir = template['save_dir'].format(task, subj, ifold)
        os.makedirs(save_dir, exist_ok=True)

        for idx in runs:
            run_id = str(idx)
            behav_path = template['behav_path'].format(subj, subj, task, run_id)
            game1ev = GAME1EV_tmp(behav_path)
            fr,fresult,angle,pic1_ap,pic2_ap,pic1_dp,pic2_dp = game1ev.label_trial_corr()
            df = df.append(pd.DataFrame({'sub_id':subj,'fightRule':fr,'fightResult':fresult,'angle':angle,
                                         'pic1_ap':pic1_ap,'pic2_ap':pic2_ap,'pic1_dp':pic1_dp,'pic2_dp':pic2_dp}),ignore_index=True)
df.to_csv('/mnt/workdir/DCM/Result/validation_trial_bias/dogfall.csv',index=False)

#%%
# get the angle distribution of dogfall trials

df = pd.read_csv('/mnt/workdir/DCM/Result/validation_trial_bias/dogfall.csv')
df_dog_fall = df[df['fightResult']==0]
angles = df_dog_fall['angle'].to_list()
plotAngleRadar(angles)
#%%

import plotly.express as px
#df_dog_fall['radians'] = df_dog_fall['angle'].apply(lambda x: (x + 180) * 3.14159 / 180)
df_dog_fall['frequency'] = df_dog_fall.groupby('angle')['angle'].transform('count')
fig = px.line_polar(df_dog_fall, r='frequency', theta=angle, line_close=False)
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, df_dog_fall['frequency'].max()]
        )),
    showlegend=False
)

fig.update_traces(fill='toself')
fig.update_layout(title='Angle Distribution')
fig.show()
