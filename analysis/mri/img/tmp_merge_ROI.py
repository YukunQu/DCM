import numpy as np
from nilearn import image


# create an atals

rois_names = ['HC','LOFC','PCC','EC','mPFC']
rois_label = [1,2,3,4,5]

# load rois
lhc = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/hippocampus/lHC_MNI152NL.nii.gz')
rhc = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/hippocampus/rHC_MNI152NL.nii.gz')
hc = image.math_img('np.logical_or(img1,img2)', img1=lhc, img2=rhc).get_fdata()

ec_img  = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/EC/juelich_EC_MNI152NL_prob.nii.gz')
ec_img = image.binarize_img(ec_img,20)
ec = ec_img.get_fdata()

mpfc = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/VMPFC/VMPFC_MNI152NL_new.nii.gz').get_fdata()

ofc1 = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/aparc/mask/lh.lateralorbitofrontal.nii.gz') #parsorbitalis
ofc2 = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/aparc/mask/rh.lateralorbitofrontal.nii.gz')
lofc = image.math_img('np.logical_or(img1,img2)', img1=ofc1, img2=ofc2).get_fdata()

pcc = image.load_img(r'/mnt/workdir/DCM/Docs/Mask/PCC/PCCk3_MNI152Nl_bin.nii.gz').get_fdata()

# sum rois
atlas = np.zeros_like(ec)
for ron, rol in zip(rois_names,rois_label):
    rn = ron.lower()
    mask = eval(rn)
    atlas[mask==1] = rol

atlas_img = image.new_img_like(ec_img,atlas)
atlas_img.to_filename(r'/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas_V2.nii.gz')


#%%
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

def check_and_modify_orientation(file_path):
    img = nib.load(file_path)
    header = img.header

    # Check and set the orientation to LPS+
    affine = img.affine
    orientation = nib.orientations.aff2axcodes(affine)
    if orientation != ('L', 'P', 'S'):
        print("The image is not LPS+ orientation")
    else:
        print("The image is LPS+ orientation")

    print(header.get_sform())


def resample_to_lps(file_path,target_img):
    img = nib.load(file_path)

    # Resample the image to LPS orientation
    resampled_img = resample_to_img(img, target_img, interpolation='nearest')

    # Zero-out the sform in the header
    resampled_img.header.set_sform(None)

    # Save the modified image
    new_file_path = f"{file_path[:-7]}_lps.nii.gz"
    nib.save(resampled_img, new_file_path)
    print(f"Resampled image saved as {new_file_path}")
    check_and_modify_orientation(new_file_path)


def zero_out_sform(file_path):
    img = nib.load(file_path)
    sform = img.header.get_sform()
    zero_sform = np.zeros_like(sform)
    img.header.set_sform(None)
    new_file_path = f"{file_path[:-7]}_zero-sform.nii.gz"
    img.to_filename(new_file_path)
    check_and_modify_orientation(new_file_path)

file_path = "/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas_lps_zero-sform.nii.gz"
target_img = '/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/mni_1mm_t1w_lps.nii.gz'
check_and_modify_orientation(file_path)
zero_out_sform(file_path)
#resample_to_lps(file_path,target_img)


#%%
file = 'DMN_atlas_lps.nii.gz'
node_names = ['HC','LOFC','PCC','EC','mPFC']
node_ids = [1,2,3,4,5]


#%%
import json

# Load the JSON file
with open('/mnt/workdir/DCM/Docs/Reference/qsirecon_atlases/atlas_config.json', 'r') as f:
    data = json.load(f)

# schaefer200x17
# Access the data in the JSON file
file_name = data['DMN_atlas']['file']
node_names = data['DMN_atlas']['node_names']
node_ids = data['DMN_atlas']['node_ids']

# Do something with the data...
#%%%
import nibabel as nib
qsi_atlas = '/mnt/workdir/DCM/Docs/Reference/qsirecon_atlases/aal116MNI_lps_mni.nii.gz'
qsi_atlas_img = nib.load(qsi_atlas)
sform = qsi_atlas_img.header.get_sform()
sform_code = qsi_atlas_img.header._structarr['sform_code']
print(sform)
print(sform_code)

my_atlas = r'/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas_lps.nii.gz'
dmn_img = nib.load(my_atlas)
print(dmn_img.header.get_sform())
qform = dmn_img.header.get_qform()
print(qform)

# zero_out the sform
dmn_img.header.set_sform(sform, int(sform_code))
dmn_img.header.set_qform(qform, int(1))
dmn_img.to_filename('/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas_lps_zero-sform.nii.gz')