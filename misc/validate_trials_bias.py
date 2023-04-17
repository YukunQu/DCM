import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

def plotAngleHist(angles):
    angles_set = set(angles)
    bins = np.linspace(-180, 180, len(angles_set))
    plt.hist(angles,bins)


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