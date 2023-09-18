import os
import pandas as pd
import subprocess


# read the subject list
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()
subjects = ''
for s in subject_list:
    subjects += s + ' '

# set the command template
cmd_tmp = 'aparcstats2table --subjects {} --hemi {} --meas {} --parc=aparc --tablefile={}'
hemis = ['lh','rh']
measures = ['thickness','volume']
output = r'/mnt/workdir/DCM/Result/analysis/structure/aparcstats2table_{}_{}.txt'

for hemi in hemis:
    for mea in measures:

        outpath = output.format(hemi,mea)
        cmd = cmd_tmp.format(subjects,hemi,mea,outpath)
        subprocess.call(cmd, shell=True)
        print(cmd)
        print('finished!')
