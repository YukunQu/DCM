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
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri>=0.5')  # look out
    subjects = data['Participant_ID'].to_list()
    spm_template = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials' \
                   r'/Setall/6fold/{}/SPM.mat'
    fmap_template = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials' \
                    r'/Setall/6fold/{}/spmF_0011.nii'
    zmap_tempalte = fmap_template.replace('spmF','test_zstats')

    for ifold in range(6,7):
        for sub_id in subjects:
            spm_file = SPMfile(spm_template.format(sub_id))
            df2 = spm_file.get_dof()

            fmap = fmap_template.format(sub_id)
            zmap = zmap_tempalte.format(sub_id)
            ftoz(fmap,2,df2,zmap)
            # convert to nii
            img = nib.load(zmap.replace('nii','nii.gz'))
            img.to_filename(zmap)
            print(f'{sub_id}')
        print("{}fold have been completed.".format(ifold))

