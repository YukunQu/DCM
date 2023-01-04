#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:02:34 2022

@author: dell
"""


# %%
def dcm2bids_helper(subjects):
    # generate the help files for dicom to bids
    import subprocess
    for subj in subjects:
        subj = str(subj).zfill(3)
        ori_dir = r'/mnt/data/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/data/DCM/tmp/{}_helper'.format(subj)
        command = r'dcm2bids_helper -d {} -o {}'.format(ori_dir, out_dir)
        print("Command:", command)
        subprocess.call(command, shell=True)


subjects = [66]
dcm2bids_helper(subjects)
# %%
import subprocess


def dcm2bids(subjects, config_file):
    for subj in subjects:
        subj = str(subj).zfill(3)
        config = config_file

        ori_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/workdir/DCM/BIDS'
        command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir, subj, config, out_dir)
        print("Command:", command)
        subprocess.call(command, shell=True)


subjects_list = [208]
config_type = 'CS'
if config_type == 'CS':
    config_ibp = r'/mnt/workdir/DCM/config/config_CS.json'  # 中科院扫描的配置文件
    dcm2bids(subjects_list, config_ibp)
elif config_type == 'Peking':
    config_pk = r'/mnt/workdir/DCM/config/config_Peking.json'  # 北大扫描的配置文件
    dcm2bids(subjects_list, config_pk)
elif config_type == 'individual':
    for sub in subjects_list:
        individual_config = r'/mnt/workdir/DCM/config/config_sub{}.json'.format(sub)
        dcm2bids([sub], individual_config)
elif config_type == 'DWI':
    config_dwi = r'/mnt/workdir/DCM/config/config_dwi.json'
    dcm2bids(subjects_list, config_dwi)
