import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch
from rsatoolbox.rdm import RDMs
from rsatoolbox.model import ModelFixed


def sub_angles(ev_files):
    runs_info = []
    for ev_file in ev_files:
        onsets = []
        conditions = []
        durations = []

        ev_info = pd.read_csv(ev_file, sep='\t')
        for group in ev_info.groupby('trial_type'):
            condition = group[0]
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        run_info = Bunch(conditions=conditions, onsets=onsets, durations=durations)
        runs_info.append(run_info)

    conditions = []
    for run_info in runs_info:
        conditions.extend(run_info.conditions)
    conditions_names = list(set(conditions))
    conditions_names.sort()
    for non_angle_reg in ['M1','decision','response']:
        conditions_names.remove(non_angle_reg)
    sub_angles_set = [float(c) for c in conditions_names]
    return sub_angles_set


def calc_rdm_angular_distance(angles_set,deg=True,scale='fine'):
    angle_num = len(angles_set)
    rdm = np.zeros(shape=(angle_num,angle_num))
    if scale == 'fine':
        for i,ia in enumerate(angles_set):
            for j,ja in enumerate(angles_set):
                angular_distance = abs(ia - ja)
                angular_distance = angular_distance % 60
                if angular_distance > 30:
                    angular_distance = 60-angular_distance
                rdm[i][j] = angular_distance
    elif scale == 'coarse':
        for i,ia in enumerate(angles_set):
            for j,ja in enumerate(angles_set):
                angular_distance = abs(ia - ja)
                angular_distance = angular_distance % 60
                if (angular_distance < 15) or (angular_distance>45):
                    angular_distance = 0
                else:
                    angular_distance = 1
                rdm[i][j] = angular_distance
    if not deg:
        rdm = np.deg2rad(rdm)
    return rdm


if __name__ == "__main__":
    ev_file_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game2/grid_rsa/{}/6fold/{}_task-game2_run-{}_events.tsv'
    savepath = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/grid_rsa_8mm/Setall/6fold/{}/{}_grid_RDM_coarse.npy'

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()
    for sub_id in subjects:
        ev_files = []
        for i in range(1,3):
            ev_files.append(ev_file_tempalte.format(sub_id,sub_id,i))
        sub_angles_set = sub_angles(ev_files)
        rdm = calc_rdm_angular_distance(sub_angles_set,scale='coarse')
        np.save(savepath.format(sub_id,sub_id),rdm)
        print("{}'s grid rdm is generated.".format(sub_id))

        #%%
        plot = False
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12,12))
            im = ax.imshow(rdm,cmap='GnBu')
            ax.set_xticks(np.arange(len(sub_angles_set)), labels=sub_angles_set)
            ax.set_yticks(np.arange(len(sub_angles_set)), labels=sub_angles_set)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")