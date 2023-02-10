import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.image import load_img,get_data
from nilearn.masking import apply_mask


#%%
# Plot eval score distribution
sub_id = 'sub-180'
#RDM_brain = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/grid_rsa_8mm/Setall/6fold/{sub_id}/rs-corr_img_coarse.nii'
RDM_brain = f'/mnt/data/DCM/result_backup/2022.11.27/game1/separate_hexagon_2phases_correct_trials/Setall/6fold/{sub_id}/ZF_0011.nii'
mask = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
eval_score = apply_mask(RDM_brain,mask)
print('mean:',eval_score.mean())
sns.displot(eval_score)
plt.plot([eval_score.mean(),eval_score.mean()],(1,5000))
#plt.title(f'Distributions of correlations for {sub_id}', size=18)
#plt.ylabel('Occurance', size=18)
#plt.xlabel('Spearmann correlation', size=18)
sns.despine()
plt.show()


#%%
# Plot eval score distribution for all subjects
import pandas as pd
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game2_fmri>0.5')
subjects = data['Participant_ID'].to_list()
subs_eval_score = []
for sub_id in subjects:
    RDM_brain = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/grid_rsa_8mm/Setall/6fold/{sub_id}/rs-corr_img_coarse.nii'
    mask = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    eval_score = apply_mask(RDM_brain,mask)
    subs_eval_score.extend(eval_score)
subs_eval_score = np.array(subs_eval_score)
#%%
print('mean:',subs_eval_score.mean())
sns.displot(subs_eval_score)
plt.plot([subs_eval_score.mean(),subs_eval_score.mean()],(1,130000))
plt.title(f'Distributions of correlations for all subjects', size=18)
plt.ylabel('Occurance', size=18)
plt.xlabel('Spearmann correlation', size=18)
sns.despine()
plt.show()