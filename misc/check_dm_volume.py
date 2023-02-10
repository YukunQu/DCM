import pandas as pd
from misc.load_spm import SPMfile
from nilearn.image import load_img

def get_sub_volumes(sub_id):
    template = r'/mnt/workdir/DCM/BIDS/{}/func/{}_task-game1_run-0{}_bold.nii.gz'
    totol_num = 0
    for i in range(1,7):
        img = load_img(template.format(sub_id,sub_id,i))
        volume_num = img.shape[-1]
        totol_num += volume_num
    return totol_num

def get_sub_rnum(sub_id):
    template = r'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/{}/SPM.mat'
    spm_file = SPMfile(template.format(sub_id))
    dm = spm_file.load_spm_dm()
    row_num = dm.shape[0]
    return row_num


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
pid = data['Participant_ID'].to_list()


for sub_id in pid:
    # get volume number of each subjects
    sub_vnum = get_sub_volumes(sub_id)
    # get row number of design matrix
    sub_rnum = get_sub_rnum(sub_id)

    if sub_vnum != sub_rnum:
        print(sub_id)