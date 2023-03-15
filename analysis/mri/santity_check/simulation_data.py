import random
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt

from nilearn.image import load_img, new_img_like,resample_to_img
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix


def simulate_grid_code_data(event_path, func_path, savepath, snr=0.8):

    omegas = range(-29,30,5) # 根据Erie 的代码重新改范围
    omega = random.choice(omegas)

    # load event
    events = pd.read_csv(event_path, sep="\t")

    decision_event = events.query("trial_type=='decision_corr'").copy()

    m2_hexagon_event = events.query("trial_type=='M2_corr'").copy()
    m2_hexagon_event['trial_type'] = 'M2_corrxcos'
    m2_hexagon_event['modulation'] = np.cos(np.deg2rad(6*(m2_hexagon_event['angle'] - omega)))

    decision_hexagon_event = events.query("trial_type=='decision_corr'").copy()
    decision_hexagon_event['trial_type'] = 'decision_corrxcos'
    decision_hexagon_event['modulation'] = np.cos(np.deg2rad(6*(decision_hexagon_event['angle'] - omega)))
    hexagon_event = pd.concat([decision_event, m2_hexagon_event, decision_hexagon_event], axis=0)

    # load func data
    func = load_img(func_path)

    # convolve HRF and generate signal series
    tr = 3
    n_scans = func.shape[-1]
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
    design_matrix = make_first_level_design_matrix(frame_times, hexagon_event, drift_model=None,
                                                   drift_order=0, hrf_model='pyspm')

    """
    ax = plot_design_matrix(design_matrix)
    plt.tight_layout()
    plt.show()
    """
    # initilize a 4D image data with noise
    dim = func.shape
    img = np.random.random(dim) * 100 * (1-snr)

    decision_signal = np.array(design_matrix['decision_corr'])
    m2_hexagon_signal = np.array(design_matrix['M2_corrxcos'])
    decision_hexagon_signal = np.array(design_matrix['decision_corrxcos'])

    # load mask
    ec_mask_path = r'/mnt/workdir/DCM/docs/Reference/Mask/EC_ROI/volume/EC-thr25-2mm.nii.gz'
    decision_mask_path = r'/mnt/workdir/DCM/docs/Reference/Mask/Park_Grid_ROI/M1r_roi.nii'
    ec_mask = load_img(ec_mask_path)
    ec_mask = resample_to_img(ec_mask,func,'nearest')
    ec_mask = ec_mask.get_fdata()

    decision_mask = load_img(decision_mask_path)
    decision_mask = decision_mask.get_fdata()

    t1_mask = load_img(r'/mnt/workdir/DCM/docs/Reference/Mask/res-02_desc-brain_mask_2mm.nii')
    t1_mask = t1_mask.get_fdata()
    # add signal into ROI
    for i in range(n_scans):
        img_slice = img[:, :, :, i]
        img_slice[t1_mask==0] = 0
        img_slice[ec_mask > 0] = m2_hexagon_signal[i] * snr * 100 + decision_hexagon_signal[i] * snr * 100
        img_slice[decision_mask > 0] = decision_signal[i] * snr * 100
        img[:, :, :, i] = img_slice

    # save data
    simulated_data = new_img_like(func, img, copy_header=True)
    simulated_data.to_filename(savepath)


if __name__ == "__main__":
    event_template = r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/separate_hexagon_2phases_correct_trials/' \
                     r'sub-180/6fold/sub-180_task-game1_run-{}_events.tsv'
    func_template = r'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap/' \
                    r'sub-180/func/sub-180_task-game1_run-0{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    save_template = r'/mnt/workdir/DCM/BIDS/derivatives/simulation_data/' \
                    r'sub-180/func/sub-180_task-game1_run-0{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_simulated.nii.gz'
    regressor_template = r'/mnt/data/DCM/derivatives/fmriprep_volume_v22_nofmap/' \
                     r'sub-180/func/sub-180_task-game1_run-{run_id}_desc-confounds_timeseries.tsv'
    for i in range(1,7):
        event_path = event_template.format(i)
        func_path = func_template.format(i)
        savepath = save_template.format(i)
        simulate_grid_code_data(event_path,func_path,savepath)
        regressor_file = regressor_template.format(i)