import time
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
import glob

## signle run ICA
signal_ica = True
if signal_ica:
    func_data = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep/sub-011/func/sub-011_task-game1_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    ica_output = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep/sub-011/func/sub-011_task-game1_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.ica'
    melodic_decomposition_command = 'melodic -i {} -o {} -v --nobet --bgthreshold=1 --tr=3 --mmthresh=0.5 --report'.format(
        func_data, ica_output)
    p = Popen(melodic_decomposition_command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout = p.stdout.read()
    stderr = p.stderr.read()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
# %%
# mulit-subject parallel ICA
start_time = time.time()
fmriprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep'

# filter subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
hp_data = data.query("(game1_acc>=0.8)and(Age>=18)")
subject_list = hp_data['Participant_ID'].to_list()

# split the subjects into many subject chunks. Each chunk includes only five subjects to prevent memory overflow.
sub_list = []
sub_set_num = 0
sub_set = []
for i, sub in enumerate(subject_list):

    sub_set.append(sub)
    if len(sub_set) == 5:
        sub_list.append(sub_set)
        sub_set = []
    elif i == (len(subject_list) - 1):
        sub_list.append(sub_set)
    else:
        continue

for subject_chunk in sub_list:
    func_list = []
    for subj_id in subject_chunk:
        func_list.extend(glob.glob(opj(fmriprep_dir,
                                       f'{subj_id}/func/{subj_id}_task-*_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_blod_smooth8.nii')))
    func_list.sort()
    output_list = [f.replace('.nii', '.ica') for f in func_list]

    melodic_decomposition_command = 'melodic -i {} -o {} -v --nobet --bgthreshold=1 --tr=3 --mmthresh=0.5 --report'
    cmds_list = [melodic_decomposition_command.format(func_data, ica_output) for func_data, ica_output in
                 zip(func_list, output_list)]
    procs_list = []
    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for outname, proc in zip(output_list, procs_list):
        proc.wait()
        print("{} finished!".format(outname))

    end_time = time.time()
    run_time = round((end_time - start_time) / 60 / 60, 2)
    print(f"Run time cost {run_time}")
