import os
import json
import pandas as pd
from scipy.stats import pearsonr

# load subject info
sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
val_sub_info = sub_info.query("game1_fmri>=0.5").sort_values('Participant_ID')
val_sub_info = val_sub_info[val_sub_info['Participant_ID'] != 'sub-089']
val_sub_info = val_sub_info[val_sub_info['Participant_ID'] != 'sub-197']
val_sub = val_sub_info['Participant_ID'].to_list()

sbm_file_dir = r'/mnt/data/DCM/HNY/Structure/SBM/Organized'
for sbm_metrics_name in os.listdir(sbm_file_dir):
    if "wholebrain" in sbm_metrics_name:
        # load vbm file
        sbm_metrics = pd.read_csv(os.path.join(sbm_file_dir, sbm_metrics_name))
        sbm_metrics = sbm_metrics.drop('Group', axis=1)

        # update sub_id for vbm meteric
        sub_list = sbm_metrics['Subject']
        sub_id_list = []
        for sub in sub_list:
            sub_id = sub.split("_")[3]
            sub_id_list.append('sub-' + sub_id)

        sbm_metrics['Subject'] = sub_id_list
        sbm_metrics = sbm_metrics[sbm_metrics['Subject'].isin(val_sub)]
        sbm_metrics = sbm_metrics.sort_values('Subject')

        beh_metric = val_sub_info['game1_acc'].to_numpy()
        metrics_names = list(sbm_metrics.columns)
        metrics_names.remove("Subject")
        result = {}
        for sm in metrics_names:
            sbm_metric = sbm_metrics[sm]
            result[sm] = pearsonr(beh_metric, sbm_metric)

        print("===================================================")

        for key, value in result.items():
            r, p = value
            #if p < (0.05/len(result)):  # look out
            print(key, ":", 'r:', str(round(r, 3)).zfill(3), 'p', round(value[1], 4))

        save_path = os.path.join('/mnt/workdir/DCM/result/structure/SBM', sbm_metrics_name.replace('csv', 'json'))
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print("===================================================")


#%%
# plot the sbm change in EC with age
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.api as sm

# load subject info
sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
val_sub_info = sub_info.query("game1_fmri>=0.5").sort_values('Participant_ID')
val_sub_info = val_sub_info[val_sub_info['Participant_ID'] != 'sub-089']
val_sub_info = val_sub_info[val_sub_info['Participant_ID'] != 'sub-197']
val_sub = val_sub_info['Participant_ID'].to_list()

# load sbm data
sbm_metrics = pd.read_csv(r'/mnt/data/DCM/HNY/Structure/SBM/Organized/stats.thickness.aparc_AllData.csv')
sbm_metrics = sbm_metrics.drop('Group', axis=1)

# update sub_id for vbm meteric
sub_list = sbm_metrics['Subject']
sub_id_list = []
for sub in sub_list:
    sub_id = sub.split("_")[3]
    sub_id_list.append('sub-' + sub_id)

sbm_metrics['Subject'] = sub_id_list
sbm_metrics = sbm_metrics[sbm_metrics['Subject'].isin(val_sub)]
sbm_metrics = sbm_metrics.sort_values('Subject')

if sbm_metrics['Subject'].to_list() == val_sub:
    print("The subjects of two groups are aligned")
else:
    print("The subjects of two groups are not aligned")

# extract EC's data
metrics_names = sbm_metrics.columns
lec_metrics_names = [mn for mn in metrics_names if 'lh_entorhinal' in mn][0]
rec_metrics_names = [mn for mn in metrics_names if 'rh_entorhinal' in mn][0]
wbrain_metric = sbm_metrics[[lec_metrics_names, rec_metrics_names]]
age = val_sub_info['Age'].to_list()
game1_acc = val_sub_info['game1_acc'].to_list()
wbrain_metric.loc[:, 'Age'] = age
wbrain_metric.loc[:, 'game1_acc'] = game1_acc
x_var = 'game1_acc'
# plot scatter figure
# left EC
r, p = pearsonr(wbrain_metric[x_var], wbrain_metric[lec_metrics_names])
print('r:',round(r,5),'p:',round(p,5))
g = sns.jointplot(x=x_var, y=lec_metrics_names, data=wbrain_metric,
                  kind="reg", truncate=False,
                  color="#d16254",
                  height=6, order=1)
g.fig.subplots_adjust(top=0.92)
if p < 0.001:
    g.fig.suptitle('r:{}  p<0.001'.format(round(r,3)),size=20)
else:
    g.fig.suptitle('r:{}, p:{}'.format(round(r,3),round(p,3)),size=20)
g.set_axis_labels(x_var,lec_metrics_names,size=20)

# right EC
r, p = pearsonr(wbrain_metric[x_var], wbrain_metric[rec_metrics_names])
print('r:',round(r,5),'p:',round(p,5))
g = sns.jointplot(x=x_var, y=rec_metrics_names, data=wbrain_metric,
                  kind="reg", truncate=False,
                  #xlim=(6, 26),  # ylim=(0, 1.05),
                  color="#d16254",
                  height=6, order=1)
g.fig.subplots_adjust(top=0.92)
if p < 0.001:
    g.fig.suptitle('r:{}  p<0.001'.format(round(r,3)),size=20)
else:
    g.fig.suptitle('r:{}, p:{}'.format(round(r,3),round(p,3)),size=20)
g.set_axis_labels(x_var,rec_metrics_names,size=20)


#%%
X = wbrain_metric[[rec_metrics_names, 'Age']]
Y = wbrain_metric['game1_acc']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
model_summary = model.summary()
print(model_summary)


#%%
# plot the whole brain's Cortexvol change with age
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.api as sm

# load subject info
sub_info = pd.read_csv(r'/mnt/workdir/DCM/BIDS/participants.tsv', sep='\t')
val_sub_info = sub_info.query("game1_fmri>=0.5").sort_values('Participant_ID')
val_sub_info = val_sub_info[val_sub_info['Participant_ID'] != 'sub-089']
val_sub_info = val_sub_info[val_sub_info['Participant_ID'] != 'sub-197']
val_sub = val_sub_info['Participant_ID'].to_list()

# load sbm data
sbm_metrics = pd.read_csv(r'/mnt/data/DCM/HNY/Structure/SBM/Organized/stats.wholebrain_AllData.csv')
sbm_metrics = sbm_metrics.drop('Group', axis=1)

# update sub_id for vbm meteric
sub_list = sbm_metrics['Subject']
sub_id_list = []
for sub in sub_list:
    sub_id = sub.split("_")[3]
    sub_id_list.append('sub-' + sub_id)

sbm_metrics['Subject'] = sub_id_list
sbm_metrics = sbm_metrics[sbm_metrics['Subject'].isin(val_sub)]
sbm_metrics = sbm_metrics.sort_values('Subject')

if sbm_metrics['Subject'].to_list() == val_sub:
    print("The subjects of two groups are aligned")
else:
    print("The subjects of two groups are not aligned")


# extract EC's data
metrics_names = sbm_metrics.columns
wbrain_metrics_names = [mn for mn in metrics_names if 'CortexVol' in mn][0]
wbrain_metric = sbm_metrics[[wbrain_metrics_names]]
age = val_sub_info['Age'].to_list()
game1_acc = val_sub_info['game1_acc'].to_list()
wbrain_metric.loc[:, 'Age'] = age
wbrain_metric.loc[:, 'game1_acc'] = game1_acc
x_var = 'Age'
# plot scatter figure
# left EC
r, p = pearsonr(wbrain_metric[x_var], wbrain_metric[wbrain_metrics_names])
print('r:',round(r,5),'p:',round(p,5))
g = sns.jointplot(x=x_var, y=wbrain_metrics_names, data=wbrain_metric,
                  kind="reg", truncate=False,
                  color="#d16254",
                  height=6, order=1)
g.fig.subplots_adjust(top=0.92)
if p < 0.001:
    g.fig.suptitle('r:{}  p<0.001'.format(round(r,3)),size=20)
else:
    g.fig.suptitle('r:{}, p:{}'.format(round(r,3),round(p,3)),size=20)
g.set_axis_labels(x_var,wbrain_metrics_names,size=20)
