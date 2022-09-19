# -*- coding: utf-8 -*-


import os
import mne 


def check_meg(data_fname):
    f = os.path.basename(data_fname)
    try:
        if '.fif' in f:
            data = mne.io.read_raw_fif(data_fname,allow_maxshield=True)
        elif '.ds' in f:
            raw = mne.io.read_raw_ctf(data_fname)
    except :
        print("Error: {} file can't be read!".format(f))
        return 0
    else:
        return 1


def check_subs_meg(sub_list,data_dir):
    info_list = []
    for sub in sub_list:
        print(f"--------------{sub} start!-------------------------------------")
        ndata_dir = os.path.join(data_dir,sub,'NeuroData','MEG')
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
    return info_list

if __name__ == "__main__":
    # load subject file and check it
    data_dir = r'/mnt/data/Sourcedata/DCM'
    sub_list = ['sub_'+str(i).zfill(3) for i in range(82,120)]
    sub_list = ['sub_145', 'sub_146', 'sub_156','sub_157','sub_168',
                'sub_169', 'sub_170', 'sub_183', 'sub_184',
                'sub_185','sub_186','sub_187',
                'sub_189','sub_190']
    log = check_subs_meg(sub_list,data_dir)
    for info in log:
        print(info)