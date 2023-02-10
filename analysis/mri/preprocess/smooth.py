# smoothing the preprocessed data for subsequent analysis
import os
import time
import pandas as pd
from os.path import join as pjoin

from nipype import SelectFiles
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import DataSink
from nipype.interfaces import spm

from nipype import Node, MapNode, Workflow
from nipype.interfaces.utility import Function,IdentityInterface
from nipype.interfaces.spm import Smooth


from niflow.nipype1.workflows.fmri.fsl.preprocess import create_susan_smooth
from nipype.interfaces.io import ExportFile

def smoothing_susan(in_file,fwhm,mask,out_file):
    smooth = create_susan_smooth()
    smooth.inputs.inputnode.in_files = in_file
    smooth.inputs.inputnode.fwhm = fwhm
    smooth.inputs.inputnode.mask_file = mask

    exflie = Node(ExportFile(out_file=out_file),name='exfile')
    smooth.connect()
    smooth.run()


in_file = r'/mnt/data/DCM/derivatives/fmriprep_volume_v22/sub-011/func/sub-011_task-game1_run-01_space-T1w_desc-preproc_bold.nii.gz'
fwhm = 6
mask = r'/mnt/data/DCM/derivatives/fmriprep_volume_v22/sub-011/anat/sub-011_desc-brain_mask.nii.gz'
smoothing_susan(in_file,fwhm,mask,'')


if __name__ == "__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    data_root = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep'
    templates = {'func':'{subj_id}/func/{subj_id}_task-*_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'}
    #smooth_data(data_root=data_root,subject_list=pid,templates=templates)