import json
import numpy as np
from nilearn.image import load_img, get_data, new_img_like
from analysis.mri.img.FtoZ_transformation import p_to_z

def load_json_file(input_file):
    with open(input_file, 'r') as f:
        ldict = json.load(f)
    return ldict


# load atlas
atlas_img = load_img(r'/mnt/workdir/DCM/Docs/Mask/aparc_atlas/aparc+aseg.nii.gz')

# load label-name dict
input_file_path = "/mnt/workdir/DCM/Docs/Mask/aparc_atlas/aparc_label_dict.json"
label_dict = load_json_file(input_file_path)
reversed_label_dict = {value: key for key, value in label_dict.items()}

# load correlation results
corr_file_path = r'/mnt/workdir/DCM/Result/analysis/structure/brain_dmetrics/metrics_aparc_dti_MD_AllData.json'
corr_file = load_json_file(corr_file_path)

atlas_data = atlas_img.get_fdata()
corr_data = np.zeros_like(atlas_data)
for name, corr in corr_file.items():
    if 'wm' in name:
        continue
    # for structure
    # name_list = name.split('_')
    # name = 'ctx-'+name_list[0]+'-'+name_list[1]
    # for diffusion
    label = float(reversed_label_dict[name])
    r,p = corr
    if p <= 0.01:
        corr_data[atlas_data == label] = r
    else:
        corr_data[atlas_data == label] = 0
corr_img = new_img_like(atlas_img,corr_data)
corr_img.to_filename("/mnt/workdir/DCM/Result/analysis/structure/brain_dmetrics/stats.dti_MD.aparc_AllData_p.05.nii.gz")

#%%
from nilearn import image,plotting
from nilearn import surface
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

curv_left = surface.load_surf_data(fsaverage.curv_left)
#curv_left_sign = np.sign(curv_left)
texture = surface.vol_to_surf(corr_img, fsaverage.pial_left)

#html_view = plotting.plot_img_on_surf(corr_img,surf_mesh='fsaverage5')
html_view = plotting.view_surf(fsaverage.infl_left, texture,threshold=0.0,
                              bg_map=fsaverage.sulc_left,cmap='coolwarm')
html_view.open_in_browser()


curv_right = surface.load_surf_data(fsaverage.curv_right)
#curv_left_sign = np.sign(curv_left)
texture = surface.vol_to_surf(corr_img, fsaverage.pial_right)

#html_view = plotting.plot_img_on_surf(corr_img,surf_mesh='fsaverage5')
html_view = plotting.view_surf(fsaverage.infl_right, texture,threshold=0,
                               bg_map=fsaverage.sulc_right,cmap='coolwarm')
html_view.open_in_browser()


#%%
import os
import json
import numpy as np
import nibabel as nib
from surfer import Brain

def load_json_file(input_file):
    with open(input_file, 'r') as f:
        ldict = json.load(f)
    return ldict

print(__doc__)

subject_id = "fsaverage"
hemi = "lh"
surf = "inflated"

"""
Bring up the visualization.
"""
brain = Brain(subject_id, hemi, surf, background="white")

"""
Read in the automatic parcellation of sulci and gyri.
"""
aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                          subject_id, "label",
                          hemi + ".aparc.annot")
labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

"""
Make a random vector of scalar data corresponding to a value for each region in
the parcellation.

"""
# load correlation results
corr_file_path = r'/mnt/workdir/DCM/result/structure/brain_dmetrics/metrics_aparc_dti_FA_AllData.json'
corr_file = load_json_file(corr_file_path)
corr_file_key = [k for k in corr_file.keys()]
roi_data = []
for name in names:
    name = name.decode("UTF-8")
    name = 'ctx-lh-' + name
    roi_data.append(corr_file[name][0])
roi_data = np.array(roi_data)
"""
Make a vector containing the data point at each vertex.
"""
vtx_data = roi_data[labels]

"""
Handle vertices that are not defined in the annotation.
"""
vtx_data[labels == -1] = -1

"""
Display these values on the brain. Use a sequential colormap (assuming
these data move from low to high values), and add an alpha channel so the
underlying anatomy is visible.
"""
brain.add_data(vtx_data, min=0,max=0.4,thresh=0, colormap="coolwarm", alpha=.8)
#%%
brain.show_view('lateral')
brain.show_view('m')
brain.show_view('rostral')
brain.show_view('caudal')
brain.show_view('ve')
brain.show_view('frontal')
brain.show_view('par')
brain.show_view('dor')
