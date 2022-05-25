import os
import shutil


fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_ica'
target_dir = r'/mnt/workdir/DCM/tmp/fmriprep_report'
files_name = os.listdir(fmriprep_dir)
html_files = [n for n in files_name if '.html' in n]
subs_list = [n.split('.')[0] for n in html_files]
#%%
# copy html file
for html in html_files:
    ori_file = os.path.join(fmriprep_dir,html)
    target_file = os.path.join(target_dir,html)
    shutil.copy(ori_file,target_file)


# copy figure file
#%%
subs_list = ['sub-010','sub-015', 'sub-027','sub-037', 'sub-046',  'sub-076', 'sub-080']
for sub in subs_list:
    ori_file = os.path.join(fmriprep_dir,sub,'figures')
    target_file = os.path.join(target_dir,sub,'figures')
    os.mkdir(os.path.join(target_dir,sub))
    shutil.copytree(ori_file,target_file)

