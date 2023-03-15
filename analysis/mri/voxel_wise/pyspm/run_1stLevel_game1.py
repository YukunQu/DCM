import os
import pandas as pd
import nibabel as nib
from misc.load_spm import SPMfile
from analysis.mri.img.FtoZ_transformation import ftoz
from analysis.mri.voxel_wise.pyspm.firstLevel import run_firstLevel_spm

# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri>=0.5')
pid = data['Participant_ID'].to_list()

# check the existence of preprocessing file
fmriprep_dir = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume_fmapless/fmriprep'
preprocess_subs = os.listdir(fmriprep_dir)
preprocess_subs = [p for p in preprocess_subs if ('sub-' in p) and ('html' not in p)]
for p in pid:
    if p not in preprocess_subs:
        print(f"The {p} didn't have preprocess files.")

# configure parameters
configs = {'data_root': fmriprep_dir,
           'event_dir': r'/mnt/workdir/DCM/BIDS/derivatives/Events',
           'nipype_dir':'/mnt/workdir/DCM/BIDS/derivatives/Nipype',
           'task': 'game1',
           'glm_type': 'distance_spat',  # look out
           'event_name':'sub-{subj_id}_task-game1_run-{run_id}_events.tsv',
           'func_name': 'func/sub-{subj_id}_task-game1_run-{run_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_trimmed_sm8.nii',
           'regressor_name': 'func/sub-{subj_id}_task-game1_run-{run_id}_desc-confounds_timeseries_trimmed.tsv'}

# set parameter
# sets = ['odd','even']
sets = ['all']
folds = [str(i) + 'fold' for i in [6]]
runs = [1, 2, 3, 4, 5, 6]

for set_id in sets:
    for ifold in folds:
        # filter the subjects who exist.
        target_dir = rf"{configs['nipype_dir']}/game1/{configs['glm_type']}/Set{set_id}/{ifold}"
        if os.path.exists(target_dir):
            already_sub = os.listdir(target_dir)
            subject_list = [p.split('-')[-1] for p in pid if p not in already_sub]
        else:
            subject_list = [p.split('-')[-1] for p in pid]
        print("{} subjects are ready.".format(len(subject_list)))

        # run first level GLM
        run_firstLevel_spm(subject_list, set_id, runs, ifold, configs)

        # F to z statistic
        sub_list = os.listdir(target_dir)
        sub_list.sort()
        for sub in sub_list:
            sub_cmap_dir = os.path.join(target_dir, sub)
            # read df
            spm_file = SPMfile(os.path.join(sub_cmap_dir,'SPM.mat'))
            df2 = spm_file.get_dof()

            # ftoz
            cmap_list = os.listdir(sub_cmap_dir)
            for cmap in cmap_list:
                if 'spmF' in cmap:
                    fmap = os.path.join(sub_cmap_dir,cmap)
                    zmap = fmap.replace('spmF','zstats')
                    ftoz(fmap,2,df2,zmap)
                    # convert to nii
                    img = nib.load(zmap.replace('nii','nii.gz'))
                    img.to_filename(zmap)
            print(sub,"'s ftoz have been done.")