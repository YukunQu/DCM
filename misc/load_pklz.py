from nipype.utils.filemanip import loadpkl
from misc.load_spm import SPMfile
res = loadpkl(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game1/decision_grid_rsa/Setall/6fold/work_1st/_subj_id_155/level1conest/result_level1conest.pklz')
contrasts = res.inputs['contrasts']

for i in range(len(contrasts)):
    contrast_names = contrasts[i][2]
    spm = SPMfile(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/game1/decision_grid_rsa/Setall/6fold/work_1st/_subj_id_155/level1conest/SPM.mat')
    targets_index = spm.get_regs_index(contrast_names)
