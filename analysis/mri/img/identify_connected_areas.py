#%%
# Identification of connected components
import numpy as np
from scipy.ndimage import label
from nilearn import image
from nilearn.plotting import plot_roi


img = image.load_img(r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_train_hexagon_spct/mPFC_thr3.1.nii.gz')
img_data = img.get_fdata()
#img_data[img_data < 2.3] = 0
labels, _ = label(img_data)
print(np.unique(labels))
first_roi_data = ((labels == 4)|(labels==1)).astype(int)
first_roi_img = image.new_img_like(img, first_roi_data)
plot_roi(first_roi_img)
first_roi_img.to_filename(r'/mnt/data/DCM/result_backup/2023.5.14/Nilearn/game1/cv_train_hexagon_spct/OFC_thr3.1.nii.gz')