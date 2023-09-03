import os
import json
import pandas as pd
from scipy.stats import pearsonr

# load subject info
sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv',sep='\t')
val_sub_info = sub_info.query("game1_fmri>=0.5").sort_values('Participant_ID')
val_sub = val_sub_info['Participant_ID'].to_list()

vbm_file_dir = r'/mnt/data/DCM/HNY/Structure/SBM/Organized'
for vbm_file in os.listdir(vbm_file_dir):
    # load vbm file
    vbm_metrics = pd.read_csv(os.path.join(vbm_file_dir, vbm_file))
    vbm_metrics = vbm_metrics.drop('Group', axis=1)

    # update sub_id for vbm meteric
    sub_list = vbm_metrics['Subject']
    sub_id_list = []
    for sub in sub_list:
        sub_id = sub.split("_")[3]
        sub_id_list.append('sub-'+sub_id)

    vbm_metrics['Subject'] = sub_id_list
    vbm_metrics = vbm_metrics[vbm_metrics['Subject'].isin(val_sub)]
    vbm_metrics = vbm_metrics.sort_values('Subject')

    beh_metric = val_sub_info['game1_acc'].to_numpy()
    metrics_names = list(vbm_metrics.columns)
    metrics_names.remove("Subject")
    result = {}
    for vm in metrics_names:
        vbm_metric = vbm_metrics[vm]
        result[vm] = pearsonr(beh_metric,vbm_metric)

    print("------------------------------------------------")
    print(vbm_file.split(".")[0],":")
    for key,value in result.items():
        r,p = value
        if p<(0.05/len(result)):
            print(key,":",'r:',str(round(r,3)).zfill(3),'p',round(value[1],4))

    save_path = os.path.join('/mnt/workdir/DCM/result/structure/VBM',vbm_file.replace('csv','json'))
    with open(save_path, 'w') as f:
        json.dump(result, f,indent=2)