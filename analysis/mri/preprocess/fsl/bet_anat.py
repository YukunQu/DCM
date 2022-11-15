import os
import time
import pandas as pd
from os.path import join as opj
from subprocess import Popen, PIPE
import glob

# subject_list
start_time = time.time()
bids_dir = '/mnt/workdir/DCM/BIDS'

# filter subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
data = data.query("(game1_acc>=0.8)and(Age>=18)")
subject_list = data['Participant_ID'].to_list()

# split the subjects into many subject chunks. Each chunk includes only five subjects to prevent memory overflow.
sub_list = []
sub_set_num = 0
sub_set = []
for i, sub in enumerate(subject_list):

    sub_set.append(sub)
    if len(sub_set) == 20:
        sub_list.append(sub_set)
        sub_set = []
    elif i == (len(subject_list) - 1):
        sub_list.append(sub_set)
    else:
        continue

# run bet command parallelly
for subject_chunk in sub_list:
    anat_list = []
    for subj_id in subject_chunk:
        # initilize the preprocessed image directory
        sub_dir = f'/mnt/workdir/DCM/BIDS/derivatives/fsl/preprocessed/{subj_id}'
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
            os.mkdir(os.path.join(sub_dir,'anat'))
            os.mkdir(os.path.join(sub_dir,'func'))

        # grasp the subject's T1w images
        anat_list.extend(glob.glob(opj(bids_dir,f'{subj_id}/anat/{subj_id}_T1w.nii.gz')))

    anat_list.sort()

    cp_list = []
    for a in anat_list:
        a = a.replace(bids_dir, '/mnt/workdir/DCM/BIDS/derivatives/fsl/preprocessed')
        cp_list.append(a)

    output_list = []
    for a in anat_list:
        a = a.replace('T1w.nii', 'T1w_brain.nii')
        a = a.replace(bids_dir, '/mnt/workdir/DCM/BIDS/derivatives/fsl/preprocessed')
        output_list.append(a)

    cp_command = 'cp {} {}'
    bet_anat_command = 'bet {} {}'

    # cp whole-brain image
    cmds_list = [cp_command.format(raw_anat, cp_anat) for raw_anat, cp_anat in zip(anat_list,cp_list)]
    procs_list = []
    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for outname, proc in zip(output_list, procs_list):
        proc.wait()
        # debug
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        print("{} finished!".format(outname))

    # bet anatomy image
    cmds_list = [bet_anat_command.format(raw_anat, anat_brain) for raw_anat, anat_brain in zip(anat_list, output_list)]
    procs_list = []
    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for outname, proc in zip(output_list, procs_list):
        proc.wait()
        # debug
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        print("{} finished!".format(outname))


end_time = time.time()
run_time = round((end_time - start_time) / 60 / 60, 2)
print(f"Run time cost {run_time}")