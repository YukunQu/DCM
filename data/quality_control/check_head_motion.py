import json
import os

import pandas as pd
from os.path import join as pjoin


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

"""Check the head motion for each run of each subject."""

def read_quality_para(filepath, para=None):
    if para is None:
        para = ['fd_mean', 'tsnr']
    with open(filepath) as f:
        data = json.load(f)

    quality_para = {}
    for p in para:
        quality_para[p] = data[p]
    return quality_para


mriqc_dir = r'/mnt/workdir/DCM/BIDS/derivatives/mriqc'
standard = 'rigid' # look out
# get subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>0.5')  # look out
sub_list = data['Participant_ID'].to_list()
sub_age =  data['Age'].to_list()
subs_hm = pd.DataFrame(columns=['Participant_ID','task','run','fd_mean','tSNR'])
for sub,age in zip(sub_list,sub_age):
    print(f"--------------{sub} start------------")
    # read head motion for each subject
    for run_id in range(1,7):
        # game1
        filepath = pjoin(mriqc_dir,f'{sub}/func/{sub}_task-game1_run-0{run_id}_bold.json')
        if os.path.exists(filepath):
            quality_para = read_quality_para(filepath)
            fd = quality_para['fd_mean']
            tsnr = quality_para['tsnr']
        else:
            print("The",sub,f"didn't have game1 run-{run_id}")
            fd = 999
            tsnr = 0
        subs_hm = subs_hm.append({'Participant_ID':sub,'Age':age,'task':'game1','run':run_id,'fd_mean':fd,'tSNR':tsnr},
                                 ignore_index=True)

    for run_id in range(1,3):
        # game2
        filepath = pjoin(mriqc_dir,f'{sub}/func/{sub}_task-game2_run-0{run_id}_bold.json')
        if os.path.exists(filepath):
            quality_para = read_quality_para(filepath)
            fd = quality_para['fd_mean']
            tsnr = quality_para['tsnr']
        else:
            print("The",sub,f"didn't have game2 run-{run_id}")
            fd = 999
            tsnr = 0
        subs_hm = subs_hm.append({'Participant_ID':sub,'Age':age,'task':'game2','run':run_id,'fd_mean':fd,'tSNR':tsnr},
                                 ignore_index=True)

    for run_id in range(1,3):
        # rest
        filepath = pjoin(mriqc_dir,f'{sub}/func/{sub}_task-rest_run-0{run_id}_bold.json')
        if os.path.exists(filepath):
            quality_para = read_quality_para(filepath)
            fd = quality_para['fd_mean']
            tsnr = quality_para['tsnr']
        else:
            print("The",sub,f"didn't have rest run-{run_id}")
            fd = 999
            tsnr = 0
        subs_hm = subs_hm.append({'Participant_ID':sub,'Age':age,'task':'rest','run':run_id,'fd_mean':fd,'tSNR':tsnr},
                                 ignore_index=True)

quality = []
for index,row in subs_hm.iterrows():
    age = row['Age']
    fd  = row['fd_mean']
    if (age>=18)&(fd>0.2):
        quality.append('bad')
    elif (age<18)&(fd>0.5):
        quality.append('bad')
    else:
        quality.append('good')
subs_hm.loc[:,'quality'] = quality
subs_hm.to_csv(rf'/mnt/workdir/DCM/result/quality_control/{standard}/participants_data_quality.csv',index=False)

#%%

for task in ['game1','game2','rest']:

    subs_good_run_num = pd.DataFrame(columns=['Participant_ID','good_run_num'])
    for sub in sub_list:
        sub_hm = subs_hm.query(f"(task=='{task}')&(Participant_ID=='{sub}')")
        good_run_num = (sub_hm['quality']=='good').sum()
        subs_good_run_num = subs_good_run_num.append({'Participant_ID':sub,
                                                      'good_run_num':good_run_num},
                                                     ignore_index=True)
    good_run_subs_num = subs_good_run_num.value_counts(subset='good_run_num')
    # plot data quality
    good_run_subs_num.sort_index()
    good_run = good_run_subs_num.index
    nums = good_run_subs_num.to_list()

    # plot bar chart of subject's number with labels
    fig,ax = plt.subplots(figsize=(12,8))
    rects = ax.bar(good_run,nums)

    # add some text for labels, title and custom x-axis tick labels, etc
    ax.set_xlabel("Good_run",size=22)
    ax.set_ylabel("Number",size=22)
    ax.bar_label(rects,padding=3,size=22)

    fig.tight_layout()
    plt.savefig(f'/mnt/workdir/DCM/result/quality_control/{standard}/{task}_good_run_subs_num.png')
    plt.show()

    # plot the subjcet states according to the good run number of subjects
    if standard == 'rigid':
        if task =='game1':
            good_sub_list = subs_good_run_num[subs_good_run_num['good_run_num']==6]['Participant_ID'].to_list()
        else:
            good_sub_list = subs_good_run_num[subs_good_run_num['good_run_num']==2]['Participant_ID'].to_list()
    else:
        if task =='game1':
            good_sub_list = subs_good_run_num[subs_good_run_num['good_run_num']>4]['Participant_ID'].to_list()
        else:
            good_sub_list = subs_good_run_num[subs_good_run_num['good_run_num']==2]['Participant_ID'].to_list()

    good_sub_data = pd.DataFrame()
    for sub in good_sub_list:

        good_sub_data = good_sub_data.append(data[data['Participant_ID']==sub],ignore_index=True)

    good_sub_data.loc[good_sub_data.Age>18,'Age'] = 19
    ages = []
    nums = []
    for subs in good_sub_data.groupby('Age'):
        ages.append(subs[0])
        nums.append(len(subs[1]))

    # plot bar chart of subject's number with labels
    fig,ax = plt.subplots(figsize=(12,8))
    ax.plot([7.5,18.5],[10,10],color='r')
    #ax.plot([18.5,28.5],[4,4],color='r')
    rects = ax.bar(ages,nums,color='b')

    # add some text for labels, title and custom x-axis tick labels, etc
    ax.set_xlabel("Age",size=16)
    ax.set_ylabel("Number",size=16)
    ax.set_xticks(range(8,29))
    ax.set_title(f"Number of subjects per age. Total number:{len(good_sub_data)}",size=20)
    ax.bar_label(rects,padding=3,size=14)

    fig.tight_layout()
    plt.savefig(f'/mnt/workdir/DCM/result/quality_control/{standard}/{task}_subjects_count.png')
    plt.show()