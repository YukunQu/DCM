import os
import pandas as pd
from nilearn.image import mean_img
from analysis.mri.img.zscore_nii import zscore_nii
from analysis.mri.Whole_brain_analysis.secondLevel import level2nd_noPhi

task = 'game1'
glm_type = 'separate_hexagon'

contrast_list = ['ZF_0005','ZF_0006','ZT_0007','ZT_0008','ZF_0011']

participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv,sep='\t')
data = participants_data.query('game1_fmri==1')  # look out

sub_type = 'hp'
hp_data = data.query('game1_acc>=0.80')
subject_list = [p.split('_')[-1] for p in hp_data['Participant_ID'].to_list()]
print("High performance:",len(hp_data),"({} adult)".format(len(hp_data.query('Age>18'))))

zmaps = []
sets = ['Set1','Set2']
save_dir = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/separate_hexagon' \
           r'/{}/group/hp/2ndLevel/_contrast_id_ZF_0011/'
for set_id in sets:
    level2nd_noPhi(subject_list,sub_type,task,glm_type, set_id, contrast_list)
    source_dir = save_dir.format(set_id)
    file = 'spmT_0001.nii'
    prefix = 'Z'
    zscore_nii(source_dir,file,prefix)
    zmaps.append(os.path.join(source_dir, 'ZT_0001.nii'))


#  average zmap
mean_zmap = mean_img(zmaps)
mean_zmap_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/defROI/mean_zmap.nii.gz'
mean_zmap.to_filename(mean_zmap_path)
