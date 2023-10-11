import os
import json
import subprocess
import numpy as np
import pandas as pd
from nilearn import image,masking
from scipy.stats import pearsonr,zscore
import seaborn as sns
import matplotlib.pyplot as plt


#%%
# calculate FA and ADC images from dwi images and convert them to the MNI space
dwi2tensor = 'dwi2tensor {} {} -fslgrad {} {} -dkt {} -force'
tensor2metric = 'tensor2metric {} -fa {} -adc {} -dkt {} -mk {} -force'

# set workdir
qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'

# get sublist
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]
sub_list.sort()

# loop to calculate tensor and metrics
for sub_id in sub_list:
    # set input tempalte
    dwi_preproc = os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_space-T1w_desc-preproc_dwi.nii.gz')
    bvec = os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_space-T1w_desc-preproc_dwi.bvec')
    bval = os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_space-T1w_desc-preproc_dwi.bval')

    # set output tempalte
    tensor = os.path.join(qsiprep_dir, sub_id, 'metrics', f'{sub_id}_dwi_tensor.nii.gz')
    dkt_img = os.path.join(qsiprep_dir, sub_id, 'metrics', f'{sub_id}_dwi_dkt.nii.gz')

    # create directory
    if not os.path.exists(os.path.join(qsiprep_dir, sub_id, 'metrics')):
        os.makedirs(os.path.join(qsiprep_dir, sub_id, 'metrics'))

    # dwi2tensor
    cmd1 = dwi2tensor.format(dwi_preproc, tensor, bvec, bval, dkt_img)
    subprocess.call(cmd1, shell=True)

    # tensor2metric
    fa = os.path.join(qsiprep_dir, sub_id, 'metrics', f'{sub_id}_dwi_FA.nii.gz')
    adc = os.path.join(qsiprep_dir, sub_id, 'metrics', f'{sub_id}_dwi_ADC.nii.gz')
    mk = os.path.join(qsiprep_dir, sub_id, 'metrics', f'{sub_id}_dwi_MK.nii.gz')
    cmd2 = tensor2metric.format(tensor, fa, adc, dkt_img, mk)
    subprocess.call(cmd2, shell=True)

    # convert FA images from T1w space to the MNI space
    ref = '/mnt/workdir/DCM/Docs/Mask/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
    transform = os.path.join(qsiprep_dir,sub_id,'anat',f'{sub_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')

    input = fa
    output = fa.replace('_dwi_FA','_space-MNI152NLin2009cAsym_res-2_dwi_FA')
    antsAT_cmd1 = f'antsApplyTransforms --float --default-value 0 --input {input} -d 3 -e 3 --interpolation LanczosWindowedSinc --output {output} --reference-image {ref} -t {transform}'
    subprocess.call(antsAT_cmd1, shell=True)

    # convert ADC images from T1w space to the MNI space
    input = adc
    output = adc.replace('_dwi_ADC','_space-MNI152NLin2009cAsym_res-2_dwi_ADC')
    antsAT_cmd2 = f'antsApplyTransforms --float --default-value 0 --input {input} -d 3 -e 3 --interpolation LanczosWindowedSinc --output {output} --reference-image {ref} -t {transform}'
    subprocess.call(antsAT_cmd2, shell=True)

    # convert MK images from T1w space to the MNI space
    input = mk
    output = mk.replace('_dwi_MK','_space-MNI152NLin2009cAsym_res-2_dwi_MK')
    antsAT_cmd3 = f'antsApplyTransforms --float --default-value 0 --input {input} -d 3 -e 3 --interpolation LanczosWindowedSinc --output {output} --reference-image {ref} -t {transform}'
    subprocess.call(antsAT_cmd3, shell=True)
    print(sub_id, 'finished.')

#%%
# extract the corresponding metrics from ROIs
# get subject list
qsiprep_dir = '/mnt/workdir/DCM/BIDS/derivatives/qsiprep/qsiprep'
sub_list = os.listdir(qsiprep_dir)
sub_list = [sub for sub in sub_list if ('sub-' in sub) and ('html' not in sub)]
sub_list.sort()
sub_list = sub_list

# filter the bad subjects
i = 0
for sub_id in sub_list:
    fd = pd.read_csv(os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_dir-PA_confounds.tsv'), sep='\t')['framewise_displacement']
    mean_fd = np.nanmean(fd)
    if mean_fd > 0.5:
        i += 1
        print(i,sub_id, mean_fd)
        sub_list.remove(sub_id)

dti_metrics_name = ['FA','ADC','MK']

def load_label_dict_from_file(input_file, reverse=False):
    with open(input_file, 'r') as f:
        ldict = json.load(f)

    return {v: k for k, v in ldict.items()} if reverse else ldict

data_rows = []

for sub_id in sub_list:
    print(sub_id)
    atlas_img_path = '/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_atlas.nii.gz'
    atlas_img = image.load_img(atlas_img_path).get_fdata()
    atlas_label = np.unique(atlas_img)

    brain_mask_path = '/mnt/workdir/DCM/Docs/Mask/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'

    # load label-name dict
    input_file_path = "/mnt/workdir/DCM/Docs/Mask/DMN/DMN_atlas/DMN_label_dict.json"
    label_dict = load_label_dict_from_file(input_file_path)

    for dma in dti_metrics_name:
        dti_metric_file = os.path.join(qsiprep_dir, sub_id, 'dwi', f'{sub_id}_space-MNI152NLin2009cAsym_res-2_dwi_{dma}.nii.gz')
        dti_metric_img = image.load_img(dti_metric_file)

        # mask the image with brain mask
        brain_mask = image.get_data(brain_mask_path)
        dti_metric_data = dti_metric_img.get_fdata()
        dti_metric_data[brain_mask == 0] = np.nan

        # smoothing
        dti_metric_img = image.new_img_like(dti_metric_img, dti_metric_data)
        dti_metric_img = image.smooth_img(dti_metric_img, 6)
        dti_metric_data = dti_metric_img.get_fdata()

        for label, name in label_dict.items():
            label = float(label)
            if (label not in atlas_label):
                raise Warning("The atlas label is not in the image.")
            elif  label == 0:
                continue
            # get mean metrics for each subject in target ROI
            sub_roi_mmetric = np.nanmean(dti_metric_data[atlas_img == label])
            if dma == 'ADC':
                dma = 'MD'
            data_rows.append({'sub-id': sub_id, 'metrics': dma, 'roi':name,'value':sub_roi_mmetric})

metrics_dmn_AllData = pd.DataFrame(data_rows)
metrics_dmn_AllData.to_csv(r"/mnt/workdir/DCM/Result/analysis/structure/MRtrix/DMN_dti_metrics_AllData.csv",index=False)