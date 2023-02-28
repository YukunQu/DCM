# -*- coding: utf-8 -*-


import os
import mne


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
        for f in ndata_list:
            if '.fif' in f:
                fileNum += 1
                data_fname = os.path.join(ndata_dir, f)
                feedb = check_meg(data_fname)
                readFileNum += feedb
            elif '.ds' in f:
                fileNum += 1
                data_fname = os.path.join(ndata_dir, f)
                feedb = check_meg(data_fname)
                readFileNum += feedb
        if '.fif' in f:
            info_list.append('{} have {}/{} fif file can be read!'.format(sub, readFileNum, fileNum))
        elif '.ds' in f:
            info_list.append('{} have {}/{} ctf file can be read!'.format(sub, readFileNum, fileNum))
    return info_list


if __name__ == "__main__":
    # load subject file and check it
    data_dir = r'/mnt/data/DCM/sourcedata'
    #sub_list = ['sub_' + str(i).zfill(3) for i in range(214, 247)]
    # sub_list.remove('sub_209')
    sub_list = os.listdir(data_dir)
    sub_list = [s for s in sub_list if 'sub' in s]

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


