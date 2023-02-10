import os
import numpy as np
import pandas as pd


def load_ev_separate(event_path):
    event = pd.read_csv(event_path,sep='\t')
    event_condition = event.query("trial_type in ['M1','M2_corr','M2_error','decision_corr','decision_error']")

    pmod_cos = event.query("trial_type =='cos'")
    pmod_sin = event.query("trial_type =='sin'")

    # generate parametric modulation for M2
    m2_corrxcos = pmod_cos.copy()
    m2_corrxcos['trial_type'] = 'M2_corrxcos'
    m2_corrxsin = pmod_sin.copy()
    m2_corrxsin['trial_type'] = 'M2_corrxsin'

    # generate parametric modulation for decision
    cos_mod = pmod_cos['modulation'].to_list()
    sin_mod = pmod_sin['modulation'].to_list()

    decision_corrxcos = event.query("trial_type == 'decision_corr'")
    decision_corrxsin = decision_corrxcos.copy()

    decision_corrxcos = decision_corrxcos.replace('decision_corr','decision_corrxcos')
    decision_corrxsin = decision_corrxsin.replace('decision_corr','decision_corrxsin')

    decision_corrxcos.loc[:,'modulation'] = cos_mod
    decision_corrxsin.loc[:,'modulation'] = sin_mod

    event_condition = event_condition.append([m2_corrxcos,m2_corrxsin,decision_corrxcos,decision_corrxsin])
    event_condition = event_condition[['onset', 'duration', 'trial_type', 'modulation']]
    return event_condition


def generate_fsl_ev(sub_id,run_id):
    ev_file = rf'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/separate_hexagon_2phases_correct_trials/' \
              rf'{sub_id}/6fold/{sub_id}_task-game1_run-{run_id}_events.tsv'
    save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/FSL/preprocessed_data/ev'
    save_template = r'{}_game1_run-0{}_{}.txt'


    ev_info = load_ev_separate(ev_file)
    trial_tripe = ['onset','duration','modulation']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        condition_ev = group[1][trial_tripe].to_numpy()
        outpath = os.path.join(save_dir,save_template.format(sub_id,run_id,condition))
        np.savetxt(outpath,condition_ev,delimiter='  ')


# read the ev files of tsv version
sub_id = r'sub-079'
runs = range(1,7)

for run_id in runs:
    generate_fsl_ev(sub_id, run_id)