import glob
import os
import shutil

#  select target file and output filepath
fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep'
sub_list = ['sub-068', 'sub-069', 'sub-074', 'sub-076', 'sub-078', 'sub-079', 'sub-082',
            'sub-090', 'sub-092', 'sub-093', 'sub-099', 'sub-102', 'sub-106', 'sub-112',
            'sub-116']
ica_file_list = []

for sub in sub_list:
    ica_template = os.path.join(fmriprep_dir,sub,'func',f'{sub}_task-game1_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_blod_smooth8.ica')
    ica_file_list.extend(glob.glob(ica_template))

# set output
output_dir = r'/media/dell/421A-2172/ICA'
output_list = []
for f in ica_file_list:
    tmp  = f.split('/')[-3:]
    output_list.append(os.path.join(output_dir,tmp[0],tmp[2]))

for ori,tar in zip(ica_file_list,output_list):
    shutil.copytree(ori,tar)
    print(ori.split('/')[-1],'finished!')