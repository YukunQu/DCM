import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch
from rsatoolbox.rdm import RDMs
from rsatoolbox.model import ModelFixed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow


def draw_arrow(angle, ax, x, y, length=1, color='black'):
    rad = np.deg2rad(angle)
    dx = length * np.cos(rad)
    dy = length * np.sin(rad)
    arrow = FancyArrow(x-dx/2, y-dy/2, dx, dy, color=color, width=0.2, length_includes_head=True)
    ax.add_patch(arrow)


def get_sub_angles_nilearn(ev_files):
    """
    get angle regressors name from event files
    :param ev_files: a list of event files path
    :return:
    """
    regressors_name = []
    for ev_file in ev_files:
        ev_info = pd.read_csv(ev_file, sep='\t')
        regressors_name.extend(ev_info['trial_type'].to_list())
    angle_con_names = list(set(regressors_name))

    # remove other regressors.
    for non_angle_reg in ['M1','decision']:
        if non_angle_reg in angle_con_names:
            angle_con_names.remove(non_angle_reg)
        else:
            print(ev_files,"don't have",non_angle_reg)

    # sort angle as value
    angle_names = [int(acn.replace('angle','')) for acn in angle_con_names]
    angle_names.sort()
    return angle_names


def calc_rdm_angular_distance(angles_set,deg=True,ifold=6,scale='fine'):
    angle_num = len(angles_set)
    rdm = np.zeros(shape=(angle_num,angle_num))
    period = 360/ifold
    if scale == 'fine':
        for i,ia in enumerate(angles_set):
            for j,ja in enumerate(angles_set):
                angular_distance = abs(ia - ja)
                angular_distance = angular_distance % period
                if angular_distance > period/2:
                    angular_distance = period-angular_distance
                rdm[i][j] = angular_distance
        if not deg:
            rdm = np.deg2rad(rdm)
    elif scale == 'coarse':
        for i,ia in enumerate(angles_set):
            for j,ja in enumerate(angles_set):
                angular_distance = abs(ia - ja)
                angular_distance = angular_distance % period
                if (angular_distance < (period/4)) or (angular_distance>(period*3/4)):
                    angular_distance = 0
                else:
                    angular_distance = 1
                rdm[i][j] = angular_distance
    return rdm


if __name__ == "__main__":
    task = 'game1'
    if task == 'game1':
        runs = range(1,7)
        ev_tempalte = r'/mnt/workdir/DCM/BIDS/derivatives/Events/' \
                      r'game1/grid_rsa/{}/6fold/{}_task-game1_run-{}_events.tsv'
        savepath = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/' \
                   'game1/grid_rsa/Setall/6fold/{}/rsa/{}_grid_RDM_coarse_{}fold.npy'
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
    subjects = data['Participant_ID'].to_list()[:1]
    for sub_id in subjects:
        ev_files = []
        for i in runs:
            ev_files.append(ev_tempalte.format(sub_id,sub_id,i))
        sub_angles_set = get_sub_angles_nilearn(ev_files)

        for ifold in range(6,7):
            rdm = calc_rdm_angular_distance(sub_angles_set,ifold=ifold,scale='coarse')
            np.save(savepath.format(sub_id,sub_id,ifold),rdm)

            plot = True
            if plot:
                import matplotlib.pyplot as plt

                # Define the colors and alpha values
                colors = [(0.850980392, 0.160784314, 0.11372549, 0.5),(0.537254902, 0.537254902, 0.537254902,0.3)]  #  red to grey
                positions = [0, 1]  # corresponding positions of the colors

                # Create the custom color map
                cmap = LinearSegmentedColormap.from_list('newCMap', colors, N=256)
                plt.register_cmap(cmap=cmap)

                fig, ax = plt.subplots(figsize=(12,12))
                im = ax.imshow(rdm,cmap='newCMap')

                # Draw arrows at tick positions
                for i, angle in enumerate(sub_angles_set):
                    draw_arrow(angle, ax, i, -1.5)
                    draw_arrow(angle, ax, -1.5, i)

                ax.set_xticks(np.arange(len(sub_angles_set)),sub_angles_set)
                ax.set_yticks(np.arange(len(sub_angles_set)),sub_angles_set)

                ax.set_xlim([-5, len(sub_angles_set)+5])
                ax.set_ylim([-5, len(sub_angles_set)+5])

                plt.title("Mdoel RDM of {}fold".format(ifold),size=30)
                # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                #          rotation_mode="anchor")
                plt.savefig(r'/mnt/workdir/DCM/Result/paper/sf/sf2/RSA/RSA_model_pattern_similarity_{}fold.pdf'.format(ifold),
                            dpi=300, bbox_inches='tight', transparent=True)
        print("{}'s grid rdm is generated.".format(sub_id))