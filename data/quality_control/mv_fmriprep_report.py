import os
import shutil


fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
target_dir = r'/mnt/workdir/DCM/tmp/fmriprep_report/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

files_list = os.listdir(fmriprep_dir)
files_list.sort()
html_files = [f for f in files_list if '.html' in f]
subs_list = [n.split('.')[0] for n in html_files]

# copy html file
for html in html_files:
    print(f'--------{html}---------')
    ori_file = os.path.join(fmriprep_dir,html)
    target_file = os.path.join(target_dir,html)
    shutil.copy(ori_file,target_file)

# copy figure file
for sub in subs_list:
    print(f'--------{sub}---------')
    ori_file = os.path.join(fmriprep_dir,sub,'figures')
    target_file = os.path.join(target_dir,sub,'figures')
    os.mkdir(os.path.join(target_dir,sub))
    shutil.copytree(ori_file,target_file)
