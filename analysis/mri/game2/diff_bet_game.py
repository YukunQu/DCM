

import numpy as np
from nilearn.masking import apply_mask
from nilearn.image import load_img,resample_to_img

# load roi
hcl = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/osfstorage-archive/data/Analysis_ROI_nii/HCl_roi.nii')
hcr = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/osfstorage-archive/data/Analysis_ROI_nii/HCr_roi.nii')

ec_gird_roi = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/osfstorage-archive/data/Analysis_ROI_nii/EC_Grid_roi.nii')

ecl_roi = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/osfstorage-archive/data/Analysis_ROI_nii/ECl_roi.nii')
ecr_roi = load_img(r'/mnt/workdir/DCM/docs/Reference/Park_Grid_Coding/osfstorage-archive/data/Analysis_ROI_nii/ECr_roi.nii')

# load stats map
game1_acc_cmap = load_img(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/separate_hexagon/Setall/group/covariates/'
                          r'acc/2ndLevel/_contrast_id_ZF_0011/con_0002.nii')
game2_acc_cmap = load_img(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game2/separate_hexagon/Setall/group/covariates/'
                          r'acc/2ndLevel/_contrast_id_ZF_0011/con_0002.nii')

# apply mask
hcl,hcr,ec_gird_roi,ecl_roi,ecr_roi = [resample_to_img(mask, game1_acc_cmap,interpolation='nearest')
                                       for mask in [hcl,hcr,ec_gird_roi,ecl_roi,ecr_roi]]

# game1

rois = ['hcl','hcr','ec_gird_roi','ecl_roi','ecr_roi']
game1_means = []
game1_std = []

for roi in rois:
    betas = apply_mask(imgs=game1_acc_cmap, mask_img=eval(roi))
    betas[betas==0] = np.nan
    game1_means.append(np.nanmean(betas))
    game1_std.append(np.nanstd(betas))

#
game2_means = []
game2_std = []
for roi in ['hcl','hcr','ec_gird_roi','ecl_roi','ecr_roi']:
    betas = apply_mask(imgs=game2_acc_cmap, mask_img=eval(roi))
    betas[betas==0] = np.nan
    game2_means.append(np.nanmean(betas))
    game2_std.append(np.nanstd(betas))


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()

x = np.arange(len(rois))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width/2, game1_means, width, yerr=game1_std, label='Game1')
rects2 = ax.bar(x + width/2, game2_means, width, yerr=game2_std, label='game2')

ax.set_ylabel('Beta')
ax.set_title('Difference between game1 and game2 in HC and EC')
ax.set_xticks([x, rois])
ax.legend()


fig.tight_layout()
plt.show()