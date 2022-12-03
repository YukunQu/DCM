task = ''
glm_type = ''
set_id = ''

data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/{set_id}/6fold','sub-{subj_id}', '{contrast_id}.nii')}
container_path = f'{task}/{glm_type}/{set_id}/{ifold}/group/{sub_type}'

subject_list
contrast_1st