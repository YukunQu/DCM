import os
import shutil
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# move T1w to fsl directory
def copy_file(inf, outf):
    shutil.copy(inf, outf)
    print(outf, 'finished.')


if __name__ == "__main__":
    # set subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game2_fmri>=0.5')  # look out
    subject_list = data['Participant_ID'].to_list()
    subject_list = [s.split('-')[-1] for s in subject_list]

    # set template
    simg_in_template = os.path.join(
        '/mnt/data/DCM/derivatives/Nipype/working_dir/game2/distance_spct/Setall/6fold/work_1st/_subj_id_{}/smooth',
        'ssub-{}_task-game2_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed.nii')

    simg_out_template = os.path.join('/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{}/func/',
                                     'sub-{}_task-game2_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed_sm8.nii')

    inf_list = []
    outf_list = []
    for sub_id in subject_list:
        for run_id in range(1, 3):
            simg_in = simg_in_template.format(sub_id, sub_id, run_id)
            simg_out = simg_out_template.format(sub_id, sub_id, run_id)

            inf_list.append(simg_in)
            outf_list.append(simg_out)

    # move file parallelly
    results_list = Parallel(n_jobs=40)(delayed(copy_file)(inf, outf) for inf, outf in zip(inf_list, outf_list))