#!/usr/bin/python

# This script will generate each subjects design.fsf, but does not run it.
# It depends on your system how will launch feat

import os
import glob
import time
import pandas as pd
from os.path import join as opj

start_time = time.time()
# Set this to the directory where you'll dump all the fsf files
fsfdir = "/mnt/workdir/DCM/BIDS/derivatives/fsl/FSF"

# filter subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
data = data.query("(game1_acc>=0.8)and(Age>=18)")
subject_list = data['Participant_ID'].to_list()

# Get all the paths!
bids_dir = '/mnt/workdir/DCM/BIDS'
func_list = []
for subj_id in subject_list:
  func_list.extend(glob.glob(opj(bids_dir,
                                 f'{subj_id}/func/'
                                 f'{subj_id}_task-game1_run-*_bold.nii.gz')))
func_list.sort()


for func_path in func_list:
    file_name = func_path.split('/')[-1]
    strs = file_name.split('_')
    for s in strs:
      if 'sub-' in s:
        sub_id = s
      elif 'run-' in s:
        run_id = s
      else:
        continue
    print(sub_id,run_id,"start!")

    ntime = os.popen(f'fslnvols {func_path}').read().rstrip()
    replacements = {'SUB_ID':sub_id, 'NTPTS':ntime, 'RUNID':run_id}
    with open("%s/template_preprocess.fsf"%(fsfdir)) as infile:
      with open("%s/design_%s_%s.fsf"%(fsfdir, sub_id, run_id), 'w') as outfile:
          for line in infile:
            for src, target in replacements.items():
              line = line.replace(src, target)
            outfile.write(line)