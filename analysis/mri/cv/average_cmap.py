import os
import pandas as pd
from nilearn.image import load_img,mean_img

# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('game1_fmri==1')  # look out
pid = data['Participant_ID'].to_list()
subjects = [p.replace('_','-') for p in pid]

# set cmap directory
cmap_dir1 = r'/mnt/data/DCM/result_backup/2022.11.27/EC/alignPhi_separate_correct_trials/Seteven'
cmap_dir2 = r'/mnt/data/DCM/result_backup/2022.11.27/EC/alignPhi_separate_correct_trials/Setodd'

# set save directory
save_dir = r'/mnt/data/DCM/result_backup/2022.11.27/EC/alignPhi_separate_correct_trials/average'

# set folds and condition list
folds = [str(i)+'fold' for i in range(6,7)]
condition_list = ['con_0001.nii','con_0002.nii','con_0003.nii','con_0004.nii','con_0005.nii']

for sub in subjects:
    for ifold in folds:
        for con_file in condition_list:
            img1_path = os.path.join(cmap_dir1,ifold,sub,con_file)
            img2_path = os.path.join(cmap_dir2,ifold,sub,con_file)

            print(img1_path)
            print(img2_path)
            img1 = load_img(img1_path)
            img2 = load_img(img2_path)
            mimg = mean_img([img1,img2])
            savepath = os.path.join(save_dir,ifold,sub)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            mimg.to_filename(os.path.join(savepath,con_file))