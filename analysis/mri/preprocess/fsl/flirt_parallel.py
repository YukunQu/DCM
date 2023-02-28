import time
import pandas as pd
from subprocess import Popen, PIPE
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
data = data.query("(game1_acc>=0.8)and(Age>=18)")
subject_list = data['Participant_ID'].to_list()

subject_chunk = list_to_chunk(subject_list,30)

for sub in subject_chunk:
    inputvol = ''
    refvol = ''
    outvol = r''
    matrix = ''

    start_time = time.time()
    flirt_command = 'flirt -in {} -ref {} -applyxfm -init {} -out {} -noresample'

    cmds_list = [flirt_command.format(inv,refv,mat,outv) for inv,refv,mat,outv in zip(inputvol,refvol,matrix,outvol)]
    procs_list = []

    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for pdir, proc in zip(outvol, procs_list):
        proc.wait()
        print("{} finished!".format(pdir))