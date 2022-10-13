# 修改图像对比度
from nilearn import image
from PIL import Image, ImageEnhance
import numpy as np

img = image.load_img(r"/mnt/workdir/DCM/BIDS/sub-010/anat/sub-010_T1w.nii.gz")
data = img.get_fdata().astype(np.int64)

# Get brightness range - i.e. darkest and lightest pixels
num = np.unique(data)
num = 627
min=int(np.min(data))
max=int(np.max(data))

# Make a LUT (Look-Up Table) to translate image values
LUT=np.zeros(627,dtype=np.uint8)
LUT[min:max+1]=np.linspace(start=0,stop=num,num=(max-min)+1,endpoint=True,dtype=np.uint8)

new_data= LUT[data]
new_img = image.new_img_like(img,new_data)
new_img.to_filename(r"/mnt/workdir/DCM/BIDS/sub-010/anat/sub-010_T1w_new.nii.gz")