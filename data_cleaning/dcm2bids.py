#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:02:34 2022

@author: dell
"""

import subprocess

# dcm2bids_helper -d /mnt/workdir/DCM/sourcedata/sub_075/NeuroData/MRI/DCM075_LiuYuetian_20220417_163221911 -o /mnt/workdir/DCM/tmp

def dcm2bids(subjects,config_file):
    for subj in subjects:
        subj = str(subj).zfill(3)
        config = config_file
        
        ori_dir = r'/mnt/workdir/DCM/sourcedata/sub_{}/NeuroData/MRI'.format(subj)
        out_dir = r'/mnt/workdir/DCM/BIDS'
        command = r'dcm2bids -d {} -p {} -c {} -o {} --forceDcm2niix'.format(ori_dir,subj,config,out_dir)
        print("Command:",command)
        subprocess.call(command,shell=True)

if __name__ == "__main__":
    # Peking scaning
    subjects_pk = [76,77,78]
    config_pk = r'/mnt/workdir/DCM/config/config_Peking.json'
    #config_pk = r'/mnt/workdir/DCM/config/config_sub075.json'
    dcm2bids(subjects_pk, config_pk)

    # ibp scaning
    #subjects_ibp = [55]
    #config_ibp = r'/mnt/data/Project/DCM/config/config_CS.json'
    #dcm2bids(subjects_ibp, config_ibp)