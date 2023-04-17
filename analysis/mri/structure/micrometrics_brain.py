import os
import json
import glob
import numpy as np
import pandas as pd
from nilearn.image import load_img, get_data, new_img_like
from nilearn.masking import apply_mask

# extract the microsructural measurements of every brain areas of each subject.

def load_label_dict_from_file(input_file):
    with open(input_file, 'r') as f:
        ldict = json.load(f)
    return ldict


# load atlas
atlas_img = get_data(r'/mnt/data/DCM/tmp/aparc_atlas/aparc+aseg_MNI152_T1_2mm.nii.gz')
atlas_label = np.unique(atlas_img)

# load label-name dict
input_file_path = "/mnt/data/DCM/tmp/aparc_atlas/aparc_label_dict.json"
label_dict = load_label_dict_from_file(input_file_path)

# get all subjects' diffusion metrics file
dmetric_files = glob.glob(r'/mnt/data/DCM/HNY/Diffusion/DTI/DTI/'
                          r'Development_Cognitive_Map_*_Development_Cognitive_Map_*_DTI 参数分析_2023-03-12T18_43_40/'
                          r'smoothed_dti_FA.nii.gz')
# dmetric_files = glob.glob(r'/mnt/data/DCM/HNY/Diffusion/DTI/DTI/'
#                           r'Development_Cognitive_Map_*_Development_Cognitive_Map_*_DTI 参数分析_2023-03-12T18_43_40/'
#                           r'smoothed_dti_MD.nii.gz')
subs_id = ['sub-' + name.split('/')[-2].split('_')[3] for name in dmetric_files]
metrics_aparc_AllData = pd.DataFrame()

for sub_id,dmetric_file in zip(subs_id,dmetric_files):
    metrics_aparc_subData = {}
    metrics_aparc_subData['sub-id'] = sub_id
    sub_img = get_data(dmetric_file)
    for label, name in label_dict.items():
        label = float(label)
        if (label not in atlas_label) or (label == 0):
            continue
        sub_roi_mmetric = np.mean(sub_img[atlas_img == label])  # get mean metrics for each subject in target ROI
        metrics_aparc_subData[name] = sub_roi_mmetric
    metrics_aparc_AllData = metrics_aparc_AllData.append(metrics_aparc_subData,ignore_index=True)
metrics_aparc_AllData = metrics_aparc_AllData.set_index('sub-id')
metrics_aparc_AllData.to_csv(r"/mnt/data/DCM/HNY/Diffusion/Micro_brainareas/metrics_aparc_dti_FA_AllData.csv")


#%%
# 计算每个脑区的FA，MD，MK和推理能力/年龄的相关
import os
import json
import pandas as pd
from scipy.stats import pearsonr


dti_file_dir = r'/mnt/data/DCM/HNY/Diffusion/Micro_brainareas'
for dti_metrics_name in os.listdir(dti_file_dir):
    # load subject info
    sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
    val_sub_info = sub_info.query("game1_fmri>=0.5")
    val_sub = val_sub_info['Participant_ID'].to_list()

    # load vbm file
    dti_metrics = pd.read_csv(os.path.join(dti_file_dir, dti_metrics_name))
    sub_id_list = dti_metrics['sub-id'].to_list()

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
    dti_metrics = dti_metrics[~dti_metrics['sub-id'].isin(diff)].sort_values('sub-id')

    val_sub = val_sub_info['Participant_ID'].to_list()
    sub_id_list = dti_metrics['sub-id'].to_list()

    # recheck if the val_sub equal to sub_id_list
    if val_sub == sub_id_list:
        print("val_sub equals sub_id_list")
    else:
        print("val_sub does not equal sub_id_list")

    beh_metric = val_sub_info['game1_acc'].to_numpy()
    metrics_names = list(dti_metrics.columns)

    metrics_names.remove("sub-id")
    result = {}
    for tm in metrics_names:
        dti_metric = dti_metrics[tm]
        result[tm] = pearsonr(beh_metric, dti_metric)
    result = dict(sorted(result.items(), key=lambda x: x[1][0], reverse=True))
    print("===================================================")

    i = 0
    for key, value in result.items():
        r, p = value
        if p < (0.05/len(result)):
            i+=1
            print(i,'个,',key, ":", 'r:', str(round(r, 3)).zfill(3), 'p', round(value[1], 4))
            save_path = os.path.join('/mnt/workdir/DCM/result/structure/brain_dmetrics/report', dti_metrics_name.replace('csv', 'txt'))
            with open(save_path, 'a') as f:
                f.write(key + " : " + "\n" + 'r:' + str(round(r, 3)).zfill(3) + '   '+'p:' + str(round(value[1], 4)) + "\n")

    save_path = os.path.join('/mnt/workdir/DCM/result/structure/brain_dmetrics', dti_metrics_name.replace('csv', 'json'))
    with open(save_path, 'w') as f:
       json.dump(result, f, indent=2)
    print("===================================================")