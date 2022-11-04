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


def smooth_data(data_root,subject_list,templates,kernel=8.):
    """
    Smooth all eligible files in the destination directory
    data_root: r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep',
    subject_list = ['sub-010','sub-011'...]
    templates = {'func':{subj_id}/func/{subj_id}_task-*_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')}
    """
    # start cue
    start_time = time.time()

    # set parameters and specify which SPM to use
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # Specify input stream
    infosource = Node(IdentityInterface(fields=['subj_id']),name="infosource")
    infosource.iterables = [('subj_id', subject_list)]

    # SelectFiles - to grab the data
    selectfiles = Node(SelectFiles(templates, base_directory=data_root, sort_filelist=True),name='selectfiles')

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])
    smooth = Node(Smooth(fwhm=[kernel]*3), name="smooth")

    # Specify working directory
    working_dir = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/smooth_data'
    # Datasink - creates output folder for outputs
    datasink_dir = '/'
    for i in data_root.split('/')[1:-1]:
        datasink_dir = datasink_dir+i+'/'
    container = data_root.split('/')[-1]
    datasink = Node(DataSink(base_directory=datasink_dir),name="datasink")

    # Initiate workflow
    analysis1st = Workflow(name='work_1st', base_dir=working_dir)

    # Connect up the nodes
    analysis1st.connect([(infosource, selectfiles,  [('subj_id','subj_id')]),
                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth,     datasink,     [('smoothed_files',container)])
                         ])
    substitutions = [('_subj_id_', ''),
                     ('ssub','func/sub'),
                     ('_bold','_bold_smooth{}'.format(int(kernel)))]
    datasink.inputs.substitutions = substitutions

    # run the smoothing pipeline
    analysis1st.run('MultiProc', plugin_args={'n_procs': 80})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")

if __name__=="__main__":
    participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
    participants_data = pd.read_csv(participants_tsv, sep='\t')
    data = participants_data.query('game1_fmri==1')
    pid = data['Participant_ID'].to_list()
    data_root = r'/mnt/workdir/DCM/BIDS/derivatives/fmriprep_volume/fmriprep'
    templates = {'func':'{subj_id}/func/{subj_id}_task-*_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'}
    smooth_data(data_root=data_root,subject_list=pid,templates=templates)