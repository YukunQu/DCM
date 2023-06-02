import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch
from rsatoolbox.rdm import RDMs
from rsatoolbox.model import ModelFixed


def get_sub_pos_condition(ev_files):
    """
    get angle regressors name from event files
    :param ev_files: a list of event files path
    :return:
    """
    regressors_name = []
    total_ev = pd.DataFrame()
    for ev_file in ev_files:
        ev_info = pd.read_csv(ev_file, sep='\t')
        total_ev = total_ev.append(ev_info,ignore_index=True)
        regressors_name.extend(ev_info['trial_type'].to_list())
    pos_con_names = list(set(regressors_name))

    # remove other regressors.
    for non_angle_reg in ['error','decision_corr','decision_error']:
        if non_angle_reg in pos_con_names:
            pos_con_names.remove(non_angle_reg)
        else:
            print(ev_files,"don't have",non_angle_reg)
    # sort angle as value
    pos_con_names.sort()

    # get postition
    pos_ap = []
    pos_dp = []
    for pos in pos_con_names:
        ap = list(set(total_ev[total_ev['trial_type']==pos]['AP'].to_list()))
        dp = list(set(total_ev[total_ev['trial_type']==pos]['DP'].to_list()))
        if (len(ap)!=1) or (len(dp)!=1):
            print(pos,'have more than one AP')
        pos_ap.append(ap[0])
        pos_dp.append(dp[0])
    return pos_con_names,pos_ap,pos_dp


def calc_rdm_pos_distance(positions,pos_ap,pos_dp):
    # calculatae distance for each position pairs
    pos_num = len(positions)
    rdm = np.zeros(shape=(pos_num,pos_num))
    for i,ia in enumerate(positions):
        for j,ja in enumerate(positions):
            x_dist = pos_ap[i] - pos_ap[j]
            y_dist = pos_dp[i] - pos_dp[j]
            rdm[i][j] = np.sqrt(x_dist**2 + y_dist**2)
    return rdm


if __name__ == "__main__":
    task = 'game1'
    if task == 'game1':
        runs = range(1,7)
        ev_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/' \
                      r'game1/map_rsa/{}/6fold/{}_task-game1_run-{}_events.tsv'
        savepath = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/' \
                   'game1/map_rsa/Setall/6fold/{}/rsa/{}_map_RDM.npy'
    elif task == 'game2':
        runs = range(1,3)
        ev_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/' \
                      r'game2/grid_rsa_corr_trials/{}/6fold/{}_task-game2_run-{}_events.tsv'
        savepath = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/' \
                   'game2/grid_rsa_corr_trials/Setall/6fold/{}/rsa/{}_grid_RDM_coarse_{}fold.npy'
    else:
        raise Exception("The task name is not right.")

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()
    for sub_id in subjects:
        print(sub_id)
        ev_files = []
        for i in runs:
            ev_files.append(ev_tempalte.format(sub_id,sub_id,i))
        pos_con_names,pos_ap,pos_dp = get_sub_pos_condition(ev_files)

        for ifold in [6]:
            rdm = calc_rdm_pos_distance(pos_con_names,pos_ap,pos_dp)
            np.save(savepath.format(sub_id,sub_id,ifold),rdm)

            plot = False
            if plot:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12,12))
                im = ax.imshow(rdm,cmap='coolwarm')
                ax.set_xticks(np.arange(len(pos_con_names)), labels=pos_con_names)
                ax.set_yticks(np.arange(len(pos_con_names)), labels=pos_con_names)
                plt.title("Mdoel RDM of map".format(ifold),size=30)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
        print("{}'s grid rdm is generated.".format(sub_id))