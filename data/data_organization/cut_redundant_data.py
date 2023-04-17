import math
import os

import numpy as np
import pandas as pd
from nilearn import image
from joblib import Parallel, delayed


def cut_redundant_data(beh_file, fmri_file, confound_file):
    print("---------------------------------------------------")
    print("behavior:", beh_file)
    print("fMRI:", fmri_file)
    print("confound:", confound_file)
    # %  load behavioral data
    beh_data = pd.read_csv(beh_file)
    columns = beh_data.columns
    if 'fix_start_cue.started' in columns:
        start_time = beh_data['fix_start_cue.started'].min()
        end_time = beh_data['cue1.started'].max() + 0.2 + 3.3 + 3
    elif 'fixation.started_raw' in columns:
        start_time = beh_data['fixation.started_raw'].min() - 1
        end_time = beh_data['cue1.started_raw'].max() + 0.2 + 3.3 + 3
    elif 'fixation.started' in columns:
        start_time = beh_data['fixation.started'].min()
        end_time = beh_data['dResp.started'].max() + 0.2 + 3.3 + 3
    elif 'fixation.started_raw' in columns:
        start_time = beh_data['fixation.started_raw'].min()
        end_time = beh_data['dResp.started_raw'].max() + 0.2 + 3.3 + 3
    else:
        raise Exception("fixation timing not exist.")
    exp_duration = end_time - start_time
    exp_tr = math.ceil(exp_duration / 3)
    # load original data
    fmri_data = image.load_img(fmri_file)

    # volume number
    scan_volume = fmri_data.shape[-1]
    redundant_time = (scan_volume - exp_tr) * 3
    data = fmri_data.get_fdata()
    if redundant_time > 0:
        data_cutted = data[:, :, :, :exp_tr]
        max_tr = exp_tr
    else:
        data_cutted = data
        max_tr = scan_volume

    #  trim fmri data as the behavioral time
    fmri_data_trimmed = image.new_img_like(fmri_data, data_cutted, copy_header=True)

    # trim confound file
    #confound_df = pd.read_csv(confound_file,sep='\t')
    #confound_df_trim = confound_df.iloc[:max_tr,:].copy()

    # save the trimmed functional image
    fmri_trimmed_savepath = fmri_file.replace('desc-preproc_bold', 'desc-preproc_bold_trimmed')
    fmri_data_trimmed.to_filename(fmri_trimmed_savepath)
    #cfile_trimmed_save = confound_file.replace('confounds_timeseries.tsv','confounds_timeseries_trimmed.tsv')
    #confound_df_trim.to_csv(cfile_trimmed_save,sep='\t',index=False)
    return redundant_time


if __name__ == "__main__":
    task = 'game2'

    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'{task}_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects_list = [p.split('-')[-1] for p in pid]
    subjects_list.sort()
    print("Total subject numbers:", len(subjects_list))

    behavior_file_list = []
    fmri_file_list = []
    confound_list = []
    for sub_id in subjects_list:
        if task == 'game1':
            for run_id in range(1, 7):
                behavior_file_list.append(
                    rf'/mnt/workdir/DCM/sourcedata/sub_{sub_id}/Behaviour/fmri_task-game1/sub-{sub_id}_task-game1_run-{run_id}.csv')
                #fmri_file_list.append(
                #    rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-game1_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
                fmri_file_list.append(
                    rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-game1_run-{run_id}_space-T1w_desc-preproc_bold.nii.gz')
                confound_list.append(
                    rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-game1_run-{run_id}_desc-confounds_timeseries.tsv')
        elif task == 'game2':
            for run_id in range(1, 3):
                behavior_file_list.append(
                    rf'/mnt/workdir/DCM/sourcedata/sub_{sub_id}/Behaviour/fmri_task-game2-test/sub-{sub_id}_task-game2_run-{run_id}.csv')
                #fmri_file_list.append(
                #    rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-game2_run-{run_id}_space-T1w_desc-preproc_bold.nii.gz')
                fmri_file_list.append(
                    rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-game2_run-{run_id}_space-T1w_desc-preproc_bold.nii.gz')
                confound_list.append(
                    rf'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-game2_run-{run_id}_desc-confounds_timeseries.tsv')
        else:
            raise Exception("The task is not support.")

    # %%
    # crop the data parallelly
    results_list = Parallel(n_jobs=70)(delayed(cut_redundant_data)(bfile, fmrif,cfile)
                                       for bfile, fmrif,cfile in zip(behavior_file_list, fmri_file_list, confound_list))
    results_list = np.array(results_list)
    np.savetxt(rf"/mnt/workdir/DCM/result/crop_data/redundant_time__new+{task}.txt", results_list, fmt='%d')
