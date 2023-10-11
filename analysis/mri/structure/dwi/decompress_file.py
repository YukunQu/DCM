from zipfile import ZipFile
import os
import pandas as pd
import numpy as np

# 指定要解压缩的目录
dir_path = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsirecon/'

# 指定要解压缩的文件名模板
file_template = '{}_dir-PA_space-T1w_desc-preproc_desc-exemplarbundles_msmtconnectome.zip'

# 指定要解压缩的所有主题编号
# get subject list
qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]
sub_list.sort()

good_sub = []
bad_sub = []
# filter the bad subjects
i = 0

for sub_id in sub_list:
    fd = pd.read_csv(os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_confounds.tsv'), sep='\t')['framewise_displacement']
    mean_fd = np.nanmean(fd)
    if mean_fd > 0.5:
        i += 1
        print(i,sub_id, mean_fd)
        bad_sub.append(sub_id)
    else:
        good_sub.append(sub_id)

print(len(good_sub))


# 遍历所有主题编号
lost_sub = []
for subject in good_sub:
    # 构造zip文件的完整路径
    file_name = file_template.format(subject)
    file_path = os.path.join(dir_path, '{}'.format(subject), 'dwi', file_name)

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 解压缩文件
        with ZipFile(file_path,'r') as zipObj:
            zipObj.extractall(os.path.join(dir_path, '{}'.format(subject), 'dwi'))
        print('{} connectome file has been decompressed.'.format(subject))
    else:
        lost_sub.append(subject)

for l in lost_sub:
    print(f'{l} connectome file does not exist.')