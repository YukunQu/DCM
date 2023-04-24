import os
import shutil
import pandas as pd


def copy_files(src_base_dir, dest_base_dir, sub_list, keyword=False):
    for sub in sub_list:
        print(sub,"started!")
        src_dir = os.path.join(src_base_dir, sub, "NeuroData/MRI")
        dest_dir = os.path.join(dest_base_dir, sub, "anat")

        if not os.path.exists(src_dir):
            print(f"Source directory does not exist: {src_dir}")
            continue

        shutil.copytree(src_dir, dest_dir)

        if keyword:
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if keyword in file:
                        src_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(src_file_path, src_dir)
                        dest_file_path = os.path.join(dest_dir, relative_path)
                        dest_folder_path = os.path.dirname(dest_file_path)
                        os.makedirs(dest_folder_path, exist_ok=True)
                        shutil.copytree(src_file_path, dest_file_path)


src_base_dir = "/mnt/workdir/DCM/sourcedata"
dest_base_dir = "/mnt/data/DCM/tmp/ToLuoYao/MRI"
keyword = "csv"
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query(f'(meg_neuro==1)or(meg_neuro==0.3)')
sub_list = [s.replace("-",'_') for s in data['Participant_ID'].tolist()]

already_sub = os.listdir(dest_base_dir)
sub_list = [s for s in sub_list if s not in already_sub]

copy_files(src_base_dir, dest_base_dir, sub_list, keyword)
