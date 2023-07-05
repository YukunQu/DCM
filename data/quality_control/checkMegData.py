# -*- coding: utf-8 -*-


import os
import mne
import pandas as pd


def check_meg(data_fname):
    f = os.path.basename(data_fname)
    try:
        if '.fif' in f:
            data = mne.io.read_raw_fif(data_fname, allow_maxshield=True)
        elif '.ds' in f:
            raw = mne.io.read_raw_ctf(data_fname)
    except:
        print("Error: {} file can't be read!".format(f))
        return 0
    else:
        return 1


def check_subs_meg(sub_list, data_dir):
    info_list = []
    for sub in sub_list:
        print(f"--------------{sub} start!-------------------------------------")
        ndata_dir = os.path.join(data_dir, sub, 'NeuroData', 'MEG')
        ndata_list = os.listdir(ndata_dir)
        if len(ndata_list) == 0:
            info_list.append(f"The {sub} don't have MEG data.")
            continue
        elif len(ndata_list) == 1:
            ndata_dir = os.path.join(ndata_dir, ndata_list[0])
            ndata_list = os.listdir(ndata_dir)
        fileNum = 0
        readFileNum = 0
        print(ndata_list)
        for f in ndata_list:
            if '.fif' in f:
                fileNum += 1
                data_fname = os.path.join(ndata_dir, f)
                feedb = check_meg(data_fname)
                readFileNum += feedb
                machine = 'eketa'
            elif '.ds' in f:
                fileNum += 1
                data_fname = os.path.join(ndata_dir, f)
                feedb = check_meg(data_fname)
                readFileNum += feedb
                machine = 'ctf'
        if machine=='eketa':
                info_list.append('{} have {}/{} fif file can be read!'.format(sub, readFileNum, fileNum))
        elif machine=='ctf':
                info_list.append('{} have {}/{} ctf file can be read!'.format(sub, readFileNum, fileNum))
        else:
            raise Exception("The machine is not right.")
    return info_list


if __name__ == "__main__":
    # load subject file and check it
    data_dir = r'/mnt/data/DCM/sourcedata'

    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data#.query('game1_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    #sub_list = ['sub_'+p.split('-')[-1] for p in pid if int(p.split('-')[-1])>230]
    sub_list = ['sub_105']
    log = check_subs_meg(sub_list, data_dir)
    for info in log:
        print(info)
    
    # count the subjects number of locations
    loc_bd = 0
    loc_zky = 0
    no_data = 0
    for info in log:
        if 'fif' in info:
            loc_bd+=1
        elif 'ctf' in info:
            loc_zky+=1
        else:
            no_data+=1