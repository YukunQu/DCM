import os
import json
import pandas as pd
from scipy.stats import pearsonr


dti_file_dir = r'/mnt/data/DCM/HNY/Diffusion/DTI/Organized'
for dti_metrics_name in os.listdir(dti_file_dir):
    # load subject info
    sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
    val_sub_info = sub_info.query("game1_fmri>=0.5")
    val_sub = val_sub_info['Participant_ID'].to_list()

    # load vbm file
    dti_metrics = pd.read_csv(os.path.join(dti_file_dir, dti_metrics_name))
    sub_id_list = dti_metrics['Subject'].to_list()

    # compare sub_id_list and val_sub
    d1 = []
    d2 = []
    for s in val_sub:
        if s not in sub_id_list:d1.append(s)
    print(dti_metrics_name, ":Those validate subjects not have diffusion files:", d1)
    for s in sub_id_list:
        if s not in val_sub:d2.append(s)
    print(dti_metrics_name, ":The subjects with diffusion files are not validated:", d2)
    diff = d1+d2

    # Remove the rows from val_sub_info whose Participant_ID is present in the diff list.
    val_sub_info = val_sub_info[~val_sub_info['Participant_ID'].isin(diff)].sort_values('Participant_ID')
    dti_metrics = dti_metrics[~dti_metrics['Subject'].isin(diff)].sort_values('Subject')

    val_sub = val_sub_info['Participant_ID'].to_list()
    sub_id_list = dti_metrics['Subject'].to_list()

    # recheck if the val_sub equal to sub_id_list
    if val_sub == sub_id_list:
        print("val_sub equals sub_id_list")
    else:
        print("val_sub does not equal sub_id_list")

    beh_metric = val_sub_info['game1_acc'].to_numpy()
    metrics_names = list(dti_metrics.columns)

    metrics_names.remove("Subject")
    result = {}
    for tm in metrics_names:
        dti_metric = dti_metrics[tm]
        result[tm] = pearsonr(beh_metric, dti_metric)
    result = dict(sorted(result.items(), key=lambda x: x[1][0], reverse=True))
    print("===================================================")

    for key, value in result.items():
        r, p = value
        if p < (0.05/(len(result))):
            print(key, ":", 'r:', str(round(r, 3)).zfill(3), 'p', round(value[1], 4))
            save_path = os.path.join('/mnt/workdir/DCM/result/structure/DTI/report', dti_metrics_name.replace('csv', 'txt'))
            with open(save_path, 'a') as f:
                f.write(key + " : " + "\n" + 'r:' + str(round(r, 3)).zfill(3) + '   '+'p:' + str(round(value[1], 4)) + "\n")

    save_path = os.path.join('/mnt/workdir/DCM/result/structure/DTI', dti_metrics_name.replace('csv', 'json'))
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print("===================================================")
