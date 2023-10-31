#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:02:34 2022

@author: dell
"""
import os


def dcm2bids_helper(subjects):
    # generate the help files for dicom to bids
    import subprocess
    for subj in subjects:
        subj = str(subj).zfill(3)
        ori_dir = r'/mnt/data/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/data/DCM/tmp/Nifti/sub_{}'.format(subj)
        # ori_dir = r'/mnt/data/AIM/Development/sourcedata/MRI_T1/20230325_K207_R1{}'.format(subj)
        # out_dir = r'/mnt/data/AIM/Development/tmp/R1{}'.format(subj)
        command = r'dcm2bids_helper -d {} -o {}'.format(ori_dir, out_dir)

        # d = '/mnt/data/AIM/Development/sourcedata/sub-R1001/NeuroData/MRI/20230325_K207_R1001'
        # o = '/mnt/data/AIM/Development/sourcedata/sub-R1005/NeuroData/MRI/1001'

        d = r'/mnt/data/AIM/Development/sourcedata/MRI_T1/20230401_K128_2003'
        o = r'/mnt/data/AIM/Development/sourcedata/tmp'
        command = r'dcm2bids_helper -d {} -o {}'.format(d, o)
        print("Command:", command)
        subprocess.call(command, shell=True)


# sub_list1 = os.listdir(r'/mnt/data/DCM/tmp/ToLuoYao/Elekta')
# sub_list2 = os.listdir(r'/mnt/data/DCM/tmp/ToLuoYao/MRI')
# sub_list2 = [s.replace('-', '_') for s in sub_list2]
# sub_list = [x.split('_')[-1] for x in sub_list1 if x not in sub_list2]
sub_list = ['006']
dcm2bids_helper(sub_list)

# %%
import subprocess
import pandas as pd


def dcm2bids(subjects, config_file):
    for subj in subjects:
        subj = str(subj).zfill(3)
        config = config_file

        ori_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/workdir/DCM/BIDS'
        command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir, subj, config, out_dir)
        print("Command:", command)
        subprocess.call(command, shell=True)


participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
subjects_list = data['Participant_ID'].to_list()
subjects_list.sort()
subjects_list = [s.split("-")[-1] for s in subjects_list]
subjects_list = ['207']
config_type = 'DWI'  # key parameter
if config_type == 'CS':
    config_ibp = r'/mnt/workdir/DCM/BIDS/derivatives/config/config_CS.json'  # 中科院扫描的配置文件
    dcm2bids(subjects_list, config_ibp)
elif config_type == 'PKU':
    config_pk = r'/mnt/workdir/DCM/BIDS/derivatives/config/config_Peking.json'  # 北大扫描的配置文件
    dcm2bids(subjects_list, config_pk)
elif config_type == 'individual':
    for sub in subjects_list:
        individual_config = r'/mnt/workdir/DCM/config/config_{}.json'.format(sub)
        dcm2bids([sub], individual_config)
elif config_type == 'DWI':
    config_dwi = r'/mnt/workdir/DCM/BIDS/derivatives/config/config_dwi.json'
    dcm2bids(subjects_list, config_dwi)
elif config_type == 'sbref':
    config_dwi = r'/mnt/workdir/DCM/config/config_sbref.json'
    dcm2bids(subjects_list, config_dwi)
