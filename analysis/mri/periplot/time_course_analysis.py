import os

import numpy as np
import pandas as pd
from os.path import join
from scipy import signal
from nilearn.masking import apply_mask
from nilearn.image import load_img, clean_img, smooth_img,binarize_img
import statsmodels.api as sm
from joblib import Parallel, delayed
from analysis.mri.preprocess.fsl.preprocess_melodic import list_to_chunk


# count the trial duration of all trials
def count_duration(subj,pconfigs):
    # load event
    event_dir = pconfigs['event_dir']
    task = pconfigs['task']
    glm_type = pconfigs['glm_type']
    events_name = pconfigs['events_name']

    run_list = pconfigs['run_list']
    for run_id in run_list:
        event_path = join(event_dir, task, glm_type, f'sub-{subj}', '6fold', events_name.format(subj, run_id))
        event = pd.read_csv(event_path, sep='\t')
        M2_onset = event[event['trial_type'] == 'M1']['onset'] + event[event['trial_type'] == 'M1']['duration'] # M2 onset
        trial_onset = M2_onset.to_list()[:-1]
        trial_endset = M2_onset[1:]
        trial_duration = [e-s for s,e in zip(trial_onset, trial_endset)]
    return trial_duration


def load_data(func_file, confounds_file):
    """load data and clean data"""
    # load image
    func_img = load_img(func_file)

    # load motion
    add_reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                     'csf', 'white_matter']
    confound_factors = pd.read_csv(confounds_file, sep="\t")
    motion = confound_factors[add_reg_names]
    motion = motion.fillna(0.0)

    # clean and smooth data
    high_pass_fre = 1 / 100
    func_img = clean_img(func_img, detrend=True, standardize=False,
                         high_pass=high_pass_fre,
                         t_r=3.0,
                         confounds=motion)
    func_img = smooth_img(func_img, 8.0)
    return func_img


def extract_act(trial_onset,func_data):
    """For each trial, we extracted activity estimates in an 15 s window (75 time points),
    time-locked to 1 s before the onset of each event of interest."""

    # upsampling from 3 second resolution to 0.2 second resolution,
    num_points = int(func_data.shape[0] * (3/0.2))
    upsampled_func_data = signal.resample(func_data, num_points, axis=0)

    # extracted activity estimates in an 15 s window (75 time points),
    # time-locked to 1 s before the onset of each event of interest.
    trial_act_course = np.zeros((len(trial_onset),100))

    for i,onset in enumerate(trial_onset):
        onset = onset - 1.0  # 1.0 is the time-locked
        onset_point = int(onset/0.2)-1  # 0.2 is the upsampled time resolution, 1 is considered that index start from 0
        trial_activity_roi = upsampled_func_data[onset_point:onset_point+100,:]
        mtrial_activity = trial_activity_roi.mean(axis=1)  # average across voxels
        if len(mtrial_activity) != 100:
            print(f'The trial {i+1}/{len(trial_onset)} have not enough time points! '
                  f'Missing time points {100-len(mtrial_activity)} '
                  f'will be filled with the mean of the previous all trials.')
            # fill the missing time point with the mean of the corresponding time points of the previous all trials
            for t in range(len(mtrial_activity),100):
                corresponding_pts = trial_act_course[:i,t].mean()
                mtrial_activity = np.append(mtrial_activity,corresponding_pts)
        trial_act_course[i] = mtrial_activity
    return trial_act_course


def time_point_regression(act_course,trial_mod):
    # Apply a linear regression to each time point in activity time course
    # trial modulation is x, activity is y
    # return the beta and p value of each time point
    time_point_num = act_course.shape[1]
    regressor_num = trial_mod.shape[1]
    beta = np.zeros((time_point_num,regressor_num))
    p = np.zeros((time_point_num,regressor_num))
    f = np.zeros(time_point_num)
    p_f = np.zeros(time_point_num)
    for i in range(act_course.shape[1]):
        x = trial_mod
        x = sm.add_constant(x)  # adding a constant
        y = act_course[:,i]
        model = sm.OLS(y, x).fit()
        beta[i] = model.params[1:]
        p[i] = model.pvalues[1:]
        # # test the hypothesis that both cosine and sine coefficients are zero
        # null_hypothesis = 'cos = 0, sin = 0'
        # f_test = model.f_test(null_hypothesis)
        # f[i] = f_test.fvalue.reshape(-1)
        # p_f[i] = f_test.pvalue
    return beta,p #,f,p_f


def run_peri_event_analysis(subj,pconfigs,roi):
    print(subj,'start...')
    # load event
    event_dir = pconfigs['event_dir']
    task = pconfigs['task']
    glm_type = pconfigs['glm_type']
    events_name = pconfigs['events_name']
    func_dir = pconfigs['func_dir']
    run_list = pconfigs['run_list']
    func_name = pconfigs['func_name']
    regressor_file = pconfigs['regressor_name']
    save_dir = pconfigs['save_dir']
    os.makedirs(save_dir,exist_ok=True)

    results = pd.DataFrame(columns=['subj', 'run', 'time_point', 'beta', 'p'])
    for i, run_id in enumerate(run_list):
        func_path = join(func_dir, f'sub-{subj}',  func_name.format(subj, run_id))
        confound_file = os.path.join(func_dir, f'sub-{subj}', 'func', regressor_file.format(subj, run_id))
        # load data and clean data
        func_img = load_data(func_path, confound_file)

        # extract roi data (time points * voxels)
        func_roi_data = apply_mask(func_img, roi)
        event_path = join(event_dir, task, glm_type, f'sub-{subj}', '6fold', events_name.format(subj, run_id))
        event = pd.read_csv(event_path, sep='\t')

        # find the trial onset and extract time series
        trial_onset = event[event['trial_type'] == 'M2_corr']['onset'].to_list()
        trial_act_course = extract_act(trial_onset, func_roi_data)

        # extract trial modulation
        # sin_mod = event[event['trial_type'] == 'sin']['modulation'].to_list()[::2]
        # cos_mod = event[event['trial_type'] == 'cos']['modulation'].to_list()[::2]
        # trial_mod = pd.DataFrame({'sin': sin_mod, 'cos': cos_mod})
        distance_mod = event[event['trial_type'] == 'distance']['modulation'].to_list()
        value_mod = event[event['trial_type'] == 'value']['modulation'].to_list()
        trial_mod = pd.DataFrame({'distance': distance_mod, 'value': value_mod})
        # apply linear regression to time point
        # test code
        #has_nan = trial_mod.isna().any().any()
        #print(subj,run_id,"Contains NaN: ", has_nan)
        beta,p = time_point_regression(trial_act_course, trial_mod)

        # add to results
        for tp in range(1,len(beta)+1):
            # results = results.append({'subj': 'sub-'+subj, 'run': run_id, 'time_point': tp,
            #                           'sin_beta': beta[tp-1][0], 'sin_p': p[tp-1][0],
            #                           'cos_beta': beta[tp-1][1], 'cos_p': p[tp-1][1],
            #                           'f': f[tp-1], 'f_p': p_f[tp-1]},
            #                          ignore_index=True)
            results = results.append({'subj': 'sub-'+subj, 'run': run_id, 'time_point': tp,
                                      'distance_beta': beta[tp-1][0], 'distance_p': p[tp-1][0],
                                      'value_beta': beta[tp-1][1], 'value_p': p[tp-1][1]},
                                     ignore_index=True)
    results.to_csv(join(save_dir, f'sub-{subj}_peri_event_analysis.csv'), index=False)


if __name__ == "__main__":
    # specify subjects
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query(f'game1_fmri>=0.5')
    pid = data['Participant_ID'].to_list()
    subjects = [p.split('-')[-1] for p in pid]

    # specify config
    pconfigs = {'TR': 3.0,
                'task': 'game1','glm_type': 'distance_value_spct',
                'run_list': [1,2,3,4,5,6],
                'func_dir': r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep',
                'event_dir': r'/mnt/data/DCM/result_backup/2023.5.14/Events',
                'func_name': 'func/sub-{}_task-game1_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
                'events_name': r'sub-{}_task-game1_run-{}_events.tsv',
                'regressor_name': r'sub-{}_task-game1_run-{}_desc-confounds_timeseries.tsv',
                'save_dir':'/mnt/data/DCM/derivatives/peri_event_analysis/dmPFC_test',
                }

    roi = load_img(r'/mnt/workdir/DCM/Docs/Mask/dmPFC/dmPFC_distance.nii.gz')
    # roi = load_img(r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz')
    # # roi = binarize_img(roi,10)
    # roi = load_img(r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/hexagon_spct/EC_thr3.1.nii.gz')

    subjects_chunk = list_to_chunk(subjects,40)
    for chunk in subjects_chunk:
        Parallel(n_jobs=40)(delayed(run_peri_event_analysis)(subj,pconfigs,roi) for subj in chunk)