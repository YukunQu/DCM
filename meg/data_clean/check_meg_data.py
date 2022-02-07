# -*- coding: utf-8 -*-


import os
import mne 


def check_meg(data_fname):
    try:
        if '.fif' in f:
            data = mne.io.read_raw_fif(data_fname,allow_maxshield=True)
        elif '.ds' in f:
            raw = mne.io.read_raw_ctf(data_fname)
    except :
        fname = os.path.basename(data_fname)
        print("Error: {} file can't be read!".format(fname))
        return 0
    else:
        return 1
        
if __name__ == "__main__":
    # load subject file and check it
    data_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata'
    ndata_type = 'MEG'  # neuron data type
    
    sub_list = os.listdir(data_dir)
    sub_list.sort()
    info_list = []
    for sub in sub_list[45:55]:
        ndata_dir = os.path.join(data_dir,sub,'NeuroData',ndata_type)
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
            info_list.append('{} have {}/{} fif file can be read!'.format(sub,readFileNum,fileNum))
        elif '.ds' in f:
             info_list.append('{} have {}/{} ctf file can be read!'.format(sub,readFileNum,fileNum))
    for info in info_list:
        print(info)
