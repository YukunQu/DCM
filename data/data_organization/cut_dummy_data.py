import pandas as pd
from nilearn import image


def cut_dummy_data(beh_file,fmri_file,show=False):
    #%  load behavioral data
    beh_data = pd.read_csv(beh_file)
    start_time = beh_data['fixation.started'].min()
    end_time = beh_data['text_2.started'].max() + 10
    exp_duration = end_time - start_time
    exp_tr = round(exp_duration/3)

    #%  load original data
    fmri_data = image.load_img(fmri_file)
    data = fmri_data.get_fdata()
    data_cutted = data[:,:,:,:exp_tr]

    #  cut fmri data as the behavioral time
    fmri_data_cutted = image.new_img_like(fmri_data,data_cutted,copy_header=True)
    # show to check

    if show == True:
        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_carpet
        plot_carpet(fmri_data)
        plt.show()

        plot_carpet(fmri_data_cutted)
        plt.show()

    return fmri_data_cutted

for run_id in range(1,3):
    behavioural_data_path = rf'/mnt/workdir/DCM/sourcedata/sub_080/Behaviour/fmri_task-game2-test/sub-080_task-game2_run-{run_id}.csv'
    fmri_data = rf'/mnt/workdir/DCM/BIDS/sub-080/func/sub-080_task-game2_run-0{run_id}_bold.nii.gz'
    fmri_data_cutted = cut_dummy_data(behavioural_data_path,fmri_data)
    fmri_data_cutted.to_filename(rf'/mnt/workdir/DCM/BIDS/sub-080/func/sub-080_task-game2_run-0{run_id}_bold.nii.gz')

#%%
test_data = image.load_img(rf'/mnt/workdir/DCM/BIDS/sub-080/func/sub-080_task-game2_run-01_bold.nii.gz')