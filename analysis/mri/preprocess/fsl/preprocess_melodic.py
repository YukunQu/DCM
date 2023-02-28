import os
import time
import glob
from subprocess import Popen, PIPE


def list_to_chunk(orignal_list, chunk_volume=10):
    chunk_list = []
    chunk = []
    for i, element in enumerate(orignal_list):
        chunk.append(element)
        if len(chunk) == chunk_volume:
            chunk_list.append(chunk)
            chunk = []
        elif i == (len(orignal_list) - 1):
            chunk_list.append(chunk)
        else:
            continue
    return chunk_list


def run_feat_preprocess_parall(fsf_file_list):
    start_time = time.time()
    preprocess_command = 'feat {}'
    cmds_list = [preprocess_command.format(fsf_file) for fsf_file in fsf_file_list]
    procs_list = []
    for cmd in cmds_list:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for outname, proc in zip(fsf_file_list, procs_list):
        proc.wait()
        print("{} finished!".format(outname))

    end_time = time.time()
    run_time = round((end_time - start_time) / 60 / 60, 2)
    print(f"Run time cost {run_time}")


if __name__ == "__main__":

    """
    # all fsf in fsf dir
    fsf_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fsl/FSF/smooth_6'
    fsf_tempalte = 'design_sub-*_run-*.fsf'

    fsf_file_list = glob.glob(os.path.join(fsf_dir,fsf_tempalte))
    fsf_file_list.sort()
    """

    #  fsf file of subjects
    subject_list = [79]
    subject_list = [str(s).zfill(3) for s in subject_list]

    fsf_file_list = []
    fsf_dir = r'/mnt/workdir/DCM/BIDS/derivatives/FSL/1st_level/fsf/full_analysis'
    fsf_tempalte = 'sub-{}_run-0{}_full_analysis.fsf'
    for s in subject_list:
        for run_id in range(1, 7):
            fsf_file_list.append(os.path.join(fsf_dir, fsf_tempalte.format(s, run_id)))

    fsf_file_chunk = list_to_chunk(fsf_file_list, 30)
    for fsf_files in fsf_file_chunk:
        run_feat_preprocess_parall(fsf_files)
