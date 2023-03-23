# The script check the consistency between fmri data and behavioral data

import os
import json
import datetime
import pandas as pd


def check_consistency(subject_list, task, run_list):
    # check the consistency of fmri files and behavioural files for game

    bids_fmri_template = r'/mnt/workdir/DCM/BIDS/{}/func/{}'

    if task == 'game1':
        behavioral_template = r'/mnt/workdir/DCM/sourcedata/{}/Behaviour/fmri_task-game1/{}'
        beh_template = '{}_task-game1_run-{}.csv'
        fmri_template = '{}_task-game1_run-{}_bold.json'
    elif task == 'game2':
        behavioral_template = r'/mnt/workdir/DCM/sourcedata/{}/Behaviour/fmri_task-game2-test/{}'
        beh_template = '{}_task-game2_run-{}.csv'
        fmri_template = '{}_task-game2_run-{}_bold.json'
    else:
        raise Exception("The task-", {}, "is not correct!")

    timeline_df = pd.DataFrame(columns=['sub_id', 'task', 'run', 'behavior_time', 'fmri_time'])
    for sub in subject_list:
        if sub == 'sub-024':
            continue
        print(f'---------------{task}-{sub}-------------------')
        for run in run_list:
            # read start time from the behavioral data
            beh_name = beh_template.format(sub, run)
            beh_path = behavioral_template.format(sub.replace("-", '_'), beh_name)
            tmp_df = pd.read_csv(beh_path)
            if 'date' not in tmp_df.columns:
                print(f"The {sub} do not have 'date' in behavioral data.")
                run_time = None
            else:
                run_time = tmp_df['date'][0].split('_')[-1]
                if 'wait_key1.rt' in tmp_df.columns:
                    starttime = tmp_df['wait_key1.rt'].min()
                elif 'wait_key6.rt' in tmp_df.columns:
                    starttime = tmp_df['wait_key6.rt'].min()
                elif 'cue1.started_raw' in tmp_df.columns:
                    starttime = tmp_df['cue1.started_raw'].min()
                else:
                    print(sub, '-', run, "didn't have starttime.")
                    starttime = 0
                run_time = int(run_time[:-2]) * 3600 + int(run_time[-2:]) * 60
                run_time = run_time + starttime

            #  read scanning time from json file of fmri data
            fmri_name = fmri_template.format(sub.replace("_", '-'), str(run).zfill(2))
            fmri_path = bids_fmri_template.format(sub.replace("_", '-'), fmri_name)

            f = open(fmri_path)
            jdata = json.load(f)
            fmri_time = jdata["AcquisitionTime"].split(':')
            fmri_time = int(fmri_time[0]) * 3600 + int(fmri_time[1]) * 60 + int(float(fmri_time[2]))

            timeline_df = timeline_df.append({'sub_id': sub, 'task': task, 'run': run,
                                              'behavior_time': run_time, 'start_time': starttime,
                                              'fmri_time': fmri_time}, ignore_index=True)

    timeline_df['time_diff'] = (timeline_df['fmri_time'] - timeline_df['behavior_time']) / 60
    error_run = timeline_df.query("(time_diff>6)or(time_diff<=0)")
    return timeline_df, error_run


if __name__ == "__main__":
    # get subject list
    participant_df = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
    game1_subs = participant_df.query(f"game1_fmri>0")['Participant_ID']
    game2_subs = participant_df.query(f"game2_fmri>0")['Participant_ID']
    subject_list = ['sub-{}'.format(i) for i in range(247, 250)]
    subject_list = ['sub-209','sub-250']

    game1_timeline_df, game1_error_run = check_consistency(subject_list,'game1', range(1, 7))
    game2_timeline_df, game2_error_run = check_consistency(subject_list,'game2', range(1, 3))

    """sub-010 and sub-011 game2 are exceptional cases. 
    Because their two runs  of game2 files are identical."""
