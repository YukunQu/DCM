import json
import os

import pandas as pd
from os.path import join as pjoin

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

"""Check the head motion for each run of each subject."""


def read_mriqc_metrics(filepath, para=None):
    # read confound file
    with open(filepath) as f:
        data = json.load(f)
    quality_para = {}
    if para is None:
        para = ['fd_mean', 'tsnr']
    for p in para:
        quality_para[p] = data[p]
    return quality_para


def read_fmriprep_metrics(filepath, para=None):
    df = pd.read_csv(filepath, sep='\t').fillna(0.0)
    quality_para = {}
    if para is None:
        para = ['framewise_displacement']
    for p in para:
        quality_para[p] = df[p].mean()
    return quality_para


def read_quality_para(filepath, para=None, target='mriqc'):
    if target == 'mriqc':
        return read_mriqc_metrics(filepath, para)
    else:
        return read_fmriprep_metrics(filepath, para)


mriqc_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
mriqc_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
standard = 'loose'  # look out
# get subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
sub_list = data['Participant_ID'].to_list()
sub_age = data['Age'].to_list()
subs_hm = pd.DataFrame(columns=['Participant_ID', 'task', 'run', 'fd_mean'])

#%%
for sub, age in zip(sub_list, sub_age):
    print(f"--------------{sub} start------------")
    # read head motion for each subject
    for run_id in range(1, 7):
        # game1
        filepath = pjoin(mriqc_dir, f'{sub}/func/{sub}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv')
        if os.path.exists(filepath):
            quality_para = read_quality_para(filepath,target='fmriprep')
            fd = quality_para['framewise_displacement']
        else:
            print("The", sub, f"didn't have game1 run-{run_id}")
            fd = 999
            tsnr = 0
        subs_hm = subs_hm.append(
            {'Participant_ID': sub, 'Age': age, 'task': 'game1', 'run': run_id, 'fd_mean': fd},
            ignore_index=True)

    for run_id in range(1, 3):
        # game2
        filepath = pjoin(mriqc_dir, f'{sub}/func/{sub}_task-game2_run-{run_id}_desc-confounds_timeseries.tsv')
        if os.path.exists(filepath):
            quality_para = read_quality_para(filepath,target='fmriprep')
            fd = quality_para['framewise_displacement']
        else:
            print("The", sub, f"didn't have game2 run-{run_id}")
            fd = 999
            tsnr = 0
        subs_hm = subs_hm.append(
            {'Participant_ID': sub, 'Age': age, 'task': 'game2', 'run': run_id, 'fd_mean': fd},
            ignore_index=True)
    for run_id in range(1, 3):
        # rest
        filepath = pjoin(mriqc_dir, f'{sub}/func/{sub}_task-rest_run-{run_id}_desc-confounds_timeseries.tsv')
        if os.path.exists(filepath):
            quality_para = read_quality_para(filepath,target='fmriprep')
            fd = quality_para['framewise_displacement']
        else:
            print("The", sub, f"didn't have rest run-{run_id}")
            fd = 999
            tsnr = 0
        subs_hm = subs_hm.append(
            {'Participant_ID': sub, 'Age': age, 'task': 'rest', 'run': run_id, 'fd_mean': fd},
            ignore_index=True)
quality = []
for index, row in subs_hm.iterrows():
    age = row['Age']
    fd = row['fd_mean']
    if (age >= 18) & (fd > 0.2):
        quality.append('bad')
    elif (age < 18) & (fd > 0.5):
        quality.append('bad')
    else:
        quality.append('good')
subs_hm.loc[:, 'quality'] = quality
subs_hm.to_csv(rf'/mnt/workdir/DCM/result/quality_control/{standard}/participants_data_quality.csv', index=False)


#%%
def plot_task_info(sub_quality_data, task):
    standard = 'loose'
    sub_quality_data = sub_quality_data[sub_quality_data['task']==task]
    sub_list = list(set(sub_quality_data['Participant_ID'].to_list()))
    subs_good_run_num = pd.DataFrame(columns=['Participant_ID', 'good_run_num'])
    sub_list.sort()
    for sub in sub_list:
        sub_hm = sub_quality_data.query(f"(task=='{task}')&(Participant_ID=='{sub}')")
        good_run_num = (sub_hm['quality'] == 'good').sum()
        if (task=='game1')and(good_run_num<4):
            print(sub)
        if ((task=='rest')or(task=='game2'))and(good_run_num<2):
            print(sub)
        subs_good_run_num = subs_good_run_num.append({'Participant_ID': sub,
                                                      'good_run_num': good_run_num},
                                                     ignore_index=True)
    good_run_subs_num = subs_good_run_num.value_counts(subset='good_run_num')
    # plot data quality
    good_run_subs_num.sort_index()
    good_run = good_run_subs_num.index
    nums = good_run_subs_num.to_list()

    # plot bar chart of subject's number with labels
    fig, ax = plt.subplots(figsize=(8, 8))
    rects = ax.bar(good_run, nums)

    # add some text for labels, title and custom x-axis tick labels, etc
    ax.set_xlabel("Good_run", size=22)
    ax.set_ylabel("Number", size=22)
    ax.bar_label(rects, padding=3, size=22)

    fig.tight_layout()
    plt.savefig(f'/mnt/workdir/DCM/result/quality_control/{standard}/{task}_good_run_subs_num.png')
    plt.show()

standard='loose'
subs_hm = pd.read_csv(rf'/mnt/workdir/DCM/result/quality_control/{standard}/participants_data_quality.csv')
plot_task_info(subs_hm,'game1')
plot_task_info(subs_hm,'game2')
plot_task_info(subs_hm,'rest')