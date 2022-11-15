import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.glm.first_level import make_first_level_design_matrix


def gen_design_matrix(event_path):
    event_run = pd.read_csv(event_path,sep='\t')

    pmod_cos = event_run.query("trial_type =='cos'")
    pmod_sin = event_run.query("trial_type =='sin'")

    # generate parametric modulation for M2
    m2_corrxcos = pmod_cos.copy()
    m2_corrxcos['trial_type'] = 'M2_corrxcos'

    m2_corrxsin = pmod_sin.copy()
    m2_corrxsin['trial_type'] = 'M2_corrxsin'

    # generate parametric modulation for decision
    cos_mod = pmod_cos['modulation'].to_list()
    sin_mod = pmod_sin['modulation'].to_list()

    decision_corrxcos = event_run.query("trial_type == 'decision_corr'").copy()
    decision_corrxsin = decision_corrxcos.copy()

    decision_corrxcos['trial_type'] = 'decision_corrxcos'
    decision_corrxsin['trial_type'] = 'decision_corrxsin'

    decision_corrxcos.loc[:,'modulation'] = cos_mod
    decision_corrxsin.loc[:,'modulation'] = sin_mod

    event_condition = event_run.query("trial_type in ['M1','M2_corr','M2_error','decision_corr','decision_error']")

    event_condition = event_condition.append([m2_corrxcos,m2_corrxsin,decision_corrxcos,decision_corrxsin])
    event_condition = event_condition[['onset', 'duration', 'trial_type', 'modulation']]

    n_scans = round(event_condition['onset'].max()/3)

    tr = 3
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
    design_matrix = make_first_level_design_matrix(frame_times, event_condition, drift_model=None,
                                                   drift_order=0, hrf_model='spm')

    return design_matrix


if __name__ == "__main__":
    event_template = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/separate_hexagon_2phases_correct_trials/' \
                     r'sub-120/6fold/sub-120_task-game1_run-{}_events.tsv'

    for i in range(1,7):
        event_path = event_template.format(i)
        design_matrix = gen_design_matrix(event_path)
        design_matrix = design_matrix.drop(columns='constant')
        fix,ax = plt.subplots(figsize=(10,10))
        g= sns.heatmap(design_matrix.corr(),annot=True,vmax=1,vmin=-1,cmap='coolwarm')
        g.set_xticklabels(g.get_xticklabels(),rotation=20)
        plt.show()