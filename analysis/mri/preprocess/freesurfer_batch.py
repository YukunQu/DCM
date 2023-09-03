import os
import time
import pandas as pd
from subprocess import Popen, PIPE
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()
subject_list.remove('sub-249')
subject_chunk = list_to_chunk(subject_list,70)

#%%
command_reconall = 'recon-all -s {} -i {} -all -sd {}'
mri_template = '/mnt/workdir/DCM/BIDS/{}/anat/{}_T1w.nii.gz'
output_dir = '/mnt/workdir/DCM/BIDS/derivatives/freesurfer'

starttime = time.time()
for sub_list in subject_chunk:
    procs_list = []
    for sub_id in sub_list:
        cmd = command_reconall.format(sub_id, mri_template.format(sub_id, sub_id), output_dir)
        print(cmd)
        env = os.environ.copy()
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for sub_id, proc in zip(sub_list,procs_list):
        proc.wait()
        print("{} finished!".format(sub_id))
        stdout, stderr = proc.communicate()
        if stdout:
            print("STDOUT for {}:\n{}".format(sub_id, stdout))
        if stderr:
            print("STDERR for {}:\n{}".format(sub_id, stderr))

    endtime = time.time()
    print('总共的时间为:', round((endtime - starttime)/60/60,2), 'h')

