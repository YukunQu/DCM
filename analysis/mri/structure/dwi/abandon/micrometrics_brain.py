import os
import json
import glob
import numpy as np
import pandas as pd
from nilearn.image import load_img, get_data, new_img_like
from nilearn.masking import apply_mask

# extract the microsructural measurements of every brain areas of each subject.
def load_label_dict_from_file(input_file,reverse=False):
    with open(input_file, 'r') as f:
        ldict = json.load(f)

    if reverse:
        reversed_dict = {v: k for k, v in ldict.items()}
        return reversed_dict
    else:
        return ldict


# load atlas
atlas_img = get_data(r'/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas_mni152.nii.gz')
atlas_label = np.unique(atlas_img)

# load label-name dict
input_file_path = "/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_label_dict.json"
label_dict = load_label_dict_from_file(input_file_path)

# get all subjects' diffusion metrics file
dmetric_files = glob.glob(r'/mnt/data/DCM/HNY/Diffusion/DTI/DTI/'
                          r'Development_Cognitive_Map_*_Development_Cognitive_Map_*_DTI 参数分析_2023-03-12T18_43_40/'
                          r'smoothed_dti_RD.nii.gz')
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
    metrics_aparc_subData = pd.DataFrame.from_dict(metrics_aparc_subData,orient='index').T
    metrics_aparc_AllData = pd.concat([metrics_aparc_AllData,metrics_aparc_subData],axis=0)
metrics_aparc_AllData = metrics_aparc_AllData.sort_values('sub-id')
metrics_aparc_AllData.to_csv(r"/mnt/data/DCM/HNY/Diffusion/Micro_brainareas/metrics_DMN_dti_RD_AllData.csv",index=False)

#%%
# 计算每个脑区的FA，MD，MK和推理能力/年龄的相关
import os
import json
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import masking

# load cognitive map's representation
# load label-name dict
input_file_path = "/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_label_dict.json"
label_dict = load_label_dict_from_file(input_file_path)
atlas = image.load_img(r'')

#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/grid_rsa_corr_trials/Setall/6fold/{}/rsa/rsa_zscore_img_coarse_6fold.nii.gz'
# #cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_train_hexagon_spct/Setall/6fold/{}/zmap/hexagon_zmap.nii.gz'
# #cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_test_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_even_zmap.nii.gz'
# #cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/cv_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
# #cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/cv_test_dmPFC_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
# #cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/cv_mpfc_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/distance_spct/Setall/6fold/{}/zmap/distance_zmap.nii.gz'

# get activity in ROI

# representation_name = 'grid-like code'
# data[representation_name] = subs_mean_activity

dti_file_dir = r'/mnt/data/DCM/HNY/Diffusion/Micro_brainareas'

for dti_metrics_name in os.listdir(dti_file_dir):
    if 'DMN' not in dti_metrics_name:
        continue
    # load subject info
    sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
    val_sub_info = sub_info.query("game2_fmri>=0.5")
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

    label = ''
    mask = image.new_img_like(atlas,atlas.get_data()==label)
    subs_cmap_list = [cmap_template.format(sub_id) for sub_id in sub_id_list]
    subs_mean_activity = np.mean(masking.apply_mask(subs_cmap_list, mask),axis=1)
    beh_metric = val_sub_info['game1_acc'].to_numpy()
    metrics_names = list(dti_metrics.columns)

    metrics_names.remove("sub-id")
    result = {}

    # Create a single row of sub-figures for each dti_metric in metrics_names
    n_metrics = len(metrics_names)
    fig, axes = plt.subplots(nrows=1, ncols=n_metrics, figsize=(n_metrics * 6, 6))

    for i, tm in enumerate(metrics_names):
        dti_metric = dti_metrics[tm]
        result[tm] = pearsonr(beh_metric, dti_metric)
        # Create scatter plot and regplot on the current sub-figure (axes[i])
        yname = dti_metrics_name.split('_')[3]
        if yname == 'FA':
            color = 'lightsalmon'
        elif yname == 'MD':
            color = 'lightsteelblue'
        sns.regplot(x=beh_metric, y=dti_metric, scatter=True, color=color, ax=axes[i], scatter_kws={'s':25})

        # Set title, xlabel, and ylabel
        axes[i].set_title(f'{tm}  (r={result[tm][0]:.2f}, p={result[tm][1]:.3f})',fontsize=20)
        axes[i].set_xlabel('game1_acc',fontsize=20)
        if i == 0:
            axes[i].set_ylabel(yname,fontsize=20)
    result = dict(sorted(result.items(), key=lambda x: x[1][0], reverse=True))
    print("===================================================")

    i = 0
    for key, value in result.items():
        r, p = value
        if p < (0.05/len(result)):
            i+=1
            print(i,'个,',key, ":", 'r:', str(round(r, 3)).zfill(3), 'p', round(value[1], 4))
            save_path = os.path.join('/mnt/workdir/DCM/Result/analysis/structure/brain_dmetrics/report', dti_metrics_name.replace('.csv', '_game1.txt'))
            with open(save_path, 'a') as f:
                f.write(key + " : " + "\n" + 'r:' + str(round(r, 3)).zfill(3) + '   '+'p:' + str(round(value[1], 4)) + "\n")

    save_path = os.path.join('/mnt/workdir/DCM/Result/analysis/structure/brain_dmetrics', dti_metrics_name.replace('.csv', '_game1.json'))
    with open(save_path, 'w') as f:
       json.dump(result, f, indent=2)
    print("===================================================")

#%%

# 计算每个脑区的FA，MD 和 cognitive map's representation 的相关
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import masking
import pingouin as pg


def load_label_dict_from_file(input_file,reverse=False):
    with open(input_file, 'r') as f:
        ldict = json.load(f)

    if reverse:
        reversed_dict = {v: k for k, v in ldict.items()}
        return reversed_dict
    else:
        return ldict

# load label-name dict
input_file_path = "/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_label_dict.json"
label_dict = load_label_dict_from_file(input_file_path,True)

#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/grid_rsa_corr_trials/Setall/6fold/{}/rsa/rsa_zscore_img_coarse_6fold.nii.gz'
# #cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_train_hexagon_spct/Setall/6fold/{}/zmap/hexagon_zmap.nii.gz'
# #cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_test_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_even_zmap.nii.gz'
#cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/cv_hexagon_spct/Setall/6fold/{}/cmap/alignPhi_cmap.nii.gz'
# #cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game1/cv_test_dmPFC_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
# #cmap_template = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/cv_mpfc_hexagon_spct/Setall/6fold/{}/zmap/alignPhi_zmap.nii.gz'
cmap_template = r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game2/distance_spct/Setall/6fold/{}/zmap/distance_zmap.nii.gz'
representation_name = 'Distance code'


dti_file_dir = r'/mnt/data/DCM/HNY/Diffusion/Micro_brainareas'
for dti_metrics_name in os.listdir(dti_file_dir):
    if 'DMN' not in dti_metrics_name:
        continue
    # load subject info
    sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
    val_sub_info = sub_info.query("game2_fmri>=0.5")
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

    metrics_names = list(dti_metrics.columns)
    metrics_names.remove("sub-id")
    result = {}

    # For each ROI, calculate the correlation between structure and function
    n_metrics = len(metrics_names)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    for i, tm in enumerate(metrics_names):
        if tm != 'mPFC':
            continue
        dti_metric = dti_metrics[tm]

        # load mask
        atlas = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas.nii.gz')
        atlas_img = atlas.get_fdata()
        label = float(label_dict[tm])
        mask_img = np.zeros_like(atlas_img)
        mask_img[atlas_img==label] = 1
        mask = image.new_img_like(atlas,  mask_img)

        # extract activity
        subs_cmap_list = [cmap_template.format(sub_id) for sub_id in sub_id_list]
        subs_mean_activity = np.mean(masking.apply_mask(subs_cmap_list, mask),axis=1)
        func_metric = subs_mean_activity

        #result[tm] = pearsonr(dti_metric, func_metric)
        tmp_data = pd.DataFrame({'dti_metric':dti_metric,
                                 'func_metric':func_metric,
                                 'game1_acc':val_sub_info['game1_acc'],})
        x= pg.partial_corr(tmp_data,'Activity', 'beh_diff', covar=['Age'], method='pearson')
        yname = dti_metrics_name.split('_')[3]
        if yname == 'FA':
            color = 'lightsalmon'
        elif yname == 'MD':
            color = 'lightsteelblue'
        sns.regplot(x=func_metric, y=dti_metric, scatter=True, color=color, ax=ax, scatter_kws={'s':25})

        # Set title, xlabel, and ylabel
        ax.set_title(f'{tm}  (r={result[tm][0]:.2f}, p={result[tm][1]:.3f})',fontsize=20)
        ax.set_xlabel(representation_name,fontsize=20)
        ax.set_ylabel(yname,fontsize=20)
