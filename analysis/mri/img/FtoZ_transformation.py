import os.path
import subprocess
import pandas as pd
from misc.load_spm import SPMfile
import nibabel as nib

def t_to_p(t_statistic, df):
    from scipy.stats import t
    return t.sf(t_statistic, df)


def f_to_p(f_statistic, df1, df2):
    from scipy.stats import f
    return f.sf(f_statistic, df1, df2)


def p_to_z(p):
    from scipy.stats import norm
    return norm.ppf(1 - p)


def t_to_z(t_statistic, df):
    p = t_to_p(t_statistic, df)
    return p_to_z(p)


def f_to_z(f_statistic, df1, df2):
    p = f_to_p(f_statistic, df1, df2)
    return p_to_z(p)


def z_to_p(z):
    from scipy.stats import norm
    return 1 - norm.cdf(z)

#%%

def ftoz(fstats_map,df1,df2,zout):
    """

    :param fstats_map: the input path of F-statistic map
    :param df1: the degrees of freedom of the f-contrast ( number of t-contrasts forming the f)
                or df_{R} - df_{F}
    :param df2: the degrees of freedom of the model
                df_{F} = n - k   where  k = len(variables_F)
    :param zout: the output path of z-statistic map
    """
    # call FSL ftoz commmand
    #
    command_ftoz = r'ftoz {} {} {} -zout {}'
    command = command_ftoz.format(fstats_map,df1,df2,zout)
    print("Command:",command)
    subprocess.call(command, shell=True)



if __name__ == "__main__":
        # zscore the 1st level result

        # specify subjects
        participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
        participants_data = pd.read_csv(participants_tsv, sep='\t')
        data = participants_data.query('game1_fmri>=0.5')  # look out
        subjects = data['Participant_ID'].to_list()

        # set data_root and glm
        data_root = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/distance_spct/Setall/{}fold'

        for ifold in range(6,7):
            for sub_id in subjects:
                sub_dir = os.path.join(data_root.format(ifold),sub_id)
                # get df
                spmf_path = os.path.join(sub_dir,'SPM.mat')
                spm_file = SPMfile(spmf_path)
                df2 = spm_file.get_dof()

                con_list = os.listdir(sub_dir)
                stats_map_list = [c for c in con_list if 'spm' in c]

                for stats_map in stats_map_list:
                    fmap = os.path.join(sub_dir, stats_map)
                    zmap = fmap.replace('spm','zstats_')
                    ftoz(fmap,2,df2,zmap)
                    # convert to nii
                    img = nib.load(zmap.replace('nii','nii.gz'))
                    img.to_filename(zmap)
                print("{}'s ftoz have been completed.".format(sub_id))