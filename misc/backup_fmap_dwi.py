import os
import shutil
import pandas as pd
from os.path import join as opj


# copy data to new directory selectively
def copy_mri_data(sub_id):
    original_dir = r'/mnt/workdir/DCM/BIDS'
    output_dir = '/mnt/data//DCM/fmap_backup/'

    sourMriDir = opj(original_dir, sub_id)
    targMriDir = opj(output_dir, sub_id.replace('-', '-'))

    if not os.path.exists(targMriDir):
        os.makedirs(targMriDir)

    mri_modes = ['fmap','dwi']

    for mode in mri_modes:
        if mode in ['anat', 'dwi', 'fmap']:
            source_file_path = opj(sourMriDir, mode)
            target_file_path = opj(targMriDir, mode)
            try:
                shutil.move(source_file_path, target_file_path)
            except:
                print("The ", sub_id, "didn't have", mode)
                continue

        elif mode == 'func':
            file_list = os.listdir(opj(sourMriDir, 'func'))
            target_files = []
            for f in file_list:
                if 'rest' in f:
                    target_files.append(f)
                    # print(f)
                else:
                    continue
            for target_file in target_files:
                source_file_path = opj(sourMriDir, mode, target_file)
                target_file_path = opj(targMriDir, mode, target_file)
                if not os.path.exists(opj(targMriDir, mode)):
                    os.makedirs(opj(targMriDir, mode))
                shutil.copy(source_file_path, target_file_path)


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')

    # copy meg data
    data = participants_data.query('game1_fmri>=0.5')
    subject_list = data['Participant_ID'].to_list()
    subject_list = [s.replace('-', '-') for s in subject_list]

    subject_list = ['sub-{}'.format(i) for i in range(232,238)]
    for sub in subject_list:
        copy_mri_data(sub)