"""
An example of T1w space to the MNI space:
antsApplyTransforms --float --default-value 0  \
		--input sub-060_task-game1_run-1_space-T1w_desc-yourBOLDresults.nii.gz -d 3 -e 3 \
		--interpolation LanczosWindowedSinc \
		--output sub-060_task-game1_run-1_space-MNI152NLin2009cAsym_desc-yourBOLDresults.nii.gz \
		--reference-image $TEMPLATE_DIR/tpl-MNI152NLin2009cAsym_res-02_T1w.nii \
		-t $FMRIPREP_DIR/sub-060/anat/sub-060_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
"""

import pandas as pd
from os.path import join as pjoin
from subprocess import Popen, PIPE
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')  # look out
subject_list = data['Participant_ID'].to_list()

input, output, ref, transform = [], [], [], []
for sub in subject_list:
    for run_id in range(1, 7):
        data_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
        input.append(pjoin(data_dir,
                           fr'{sub}/fsl/{sub}_task-game1_run-{run_id}_space-T1w_desc-preproc_bold_trimmed.ica/'
                           r'filtered_func_data_clean.nii.gz'))

        output.append(pjoin(data_dir,
                            fr'{sub}/fsl/{sub}_task-game1_run-{run_id}_space-T1w_desc-preproc_bold_trimmed.ica/'
                            r'filtered_func_data_clean_space-MNI152NLin2009cAsym_res-2.nii.gz'))

        ref.append(pjoin(data_dir,
                         rf'{sub}/anat/{sub}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'))

        transform.append(pjoin(data_dir,
                               rf'{sub}/anat/{sub}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'))

        antsAT_cmd = 'antsApplyTransforms --float --default-value 0 --input {} -d 3 -e 3 --interpolation LanczosWindowedSinc ' \
                     '--output {} --reference-image {} -t {}'

        cmd_list = [antsAT_cmd.format(inv, ouv, r, t) for inv, ouv, r, t in zip(input, output, ref, transform)]

cmd_list = list_to_chunk(cmd_list, 70)
for cmd_chunk in cmd_list:
    procs_list = []
    for cmd in cmd_chunk:
        print(cmd)
        procs_list.append(Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, shell=True, close_fds=True))

    for pdir, proc in zip(output, procs_list):
        proc.wait()
        print("{} finished!".format(pdir))
