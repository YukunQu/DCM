import os
import time
from os.path import join as pjoin

from nipype import Node, MapNode, Workflow
from nipype.interfaces.utility import Function,IdentityInterface
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm import Smooth
from nipype.interfaces.spm import Level1Design
from nipype.interfaces.spm import EstimateModel
from nipype.interfaces.spm import EstimateContrast

from nipype import SelectFiles
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import DataSink
from nipype.interfaces import spm


def run_info_alignPhi_separate(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2_corr','M2_error','decision_corr','decision_error']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['alignPhi']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    if conditions == ['M1','M2_corr','M2_error','decision_corr','decision_error']:
        pmod = [None,run_pmod,None,run_pmod,None]
        orth = ['No','No','No','No','No','No']
    elif conditions == ['M1','M2_corr','decision_corr']:
        pmod = [None,run_pmod,run_pmod]
        orth = ['No','No','No']
    else:
        raise Exception("The conditions are not expected.")

    motions_df = pd.read_csv(motions_file,sep='\t')
    motion_columns = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z',
                      'csf','white_matter']
    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,
                     pmod=pmod,orth=orth,regressor_names=motion_columns,regressors=motions)
    return run_info


def firstLevel_alignPhi_separate(subject_list,set_id,runs,ifold,configs):
    # start cue
    start_time = time.time()
    print("Training set",set_id," ",ifold," start!")

    # set parameters and specify which SPM to use
    tr = 3.
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # Specify input & output stream
    infosource = Node(IdentityInterface(fields=['subj_id']),name="infosource")
    infosource.iterables = [('subj_id', subject_list)]

    data_root = configs['data_root']
    event_dir = configs['event_dir']
    task = configs['task']
    glm_type = configs['glm_type']
    func_name = configs['func_name']
    event_name = configs['event_name']
    regressor_name = configs['regressor_name']

    templates = {'func': pjoin(data_root,'sub-{subj_id}',func_name),
                 'event': pjoin(event_dir,task, glm_type, 'sub-{subj_id}', ifold, event_name),
                 'regressors':pjoin(data_root,'sub-{subj_id}',regressor_name)
                 }

    # SelectFiles - to grab the data (alternativ to DataGrabber)
    selectfiles = Node(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                       name='selectfiles')
    selectfiles.inputs.run_id = runs

    # Datasink - creates output folder for important outputs
    datasink_dir = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
    working_dir = f'/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/{task}/{glm_type}/Set{set_id}/{ifold}'
    container_path = os.path.join(task,glm_type,f'Set{set_id}')
    datasink = Node(DataSink(base_directory=datasink_dir,
                             container=container_path),
                    name="datasink")

    # Use the following DataSink output substitutions
    substitutions = [('_subj_id_', 'sub-')]
    datasink.inputs.substitutions = substitutions

    # Specify GLM contrasts
    # Condition names
    condition_names = ['M2_corrxalignPhi^1','decision_corrxalignPhi^1','M2_corr','decision_corr']

    # contrastst
    cont01 = ['m2_alignPhi',            'T', condition_names,  [1,0,0,0]]
    cont02 = ['decision_alignPhi',      'T', condition_names,  [0,1,0,0]]

    cont03 = ['alignPhidecision_cos',   'T', condition_names,  [1,1,0,0]]
    cont04 = ['m2_corr',                'T', condition_names,  [0,0,1,0]]
    cont05 = ['decision_corr',          'T', condition_names,  [0,0,0,1]]

    contrast_list = [cont01,cont02,cont03,cont04,cont05]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_alignPhi_separate),
                        name='runsinfo',
                        iterfield=['ev_file','motions_file'])

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=tr,
                                     high_pass_filter_cutoff=100.,
                                     ),
                     name='modelspec')

    mask_img = r'/mnt/workdir/DCM/docs/Mask/res-02_desc-brain_mask.nii'
    #mask_img = r'/mnt/data/Template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    # Level1Design - Generates an SPM design matrix
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0,0]}},
                                     timing_units='secs',
                                     interscan_interval=3.,
                                     model_serial_correlations='AR(1)',
                                     microtime_resolution=49,
                                     microtime_onset=24,
                                     mask_image=mask_img,
                                     flags={'mthresh':float("-inf"),
                                            'volt':1}),
                        name="level1design")

    # EstimateModel - estimate the parameters of the model
    level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level1estimate")

    # EstimateContrast - estimates contrasts
    level1conest = Node(EstimateContrast(contrasts=contrast_list),
                        name="level1conest")

    # Specify Workflow
    # Initiation of the 1st-level analysis workflow
    analysis1st = Workflow(name='work_1st', base_dir=working_dir)

    # Connect up the 1st-level analysis components
    analysis1st.connect([(infosource, selectfiles,  [('subj_id','subj_id')]),
                         (selectfiles, runs_prep,   [('event','ev_file'),
                                                     ('regressors','motions_file')]),
                         (runs_prep, modelspec,     [('run_info','subject_info')]),

                         (selectfiles, gunzip_func,  [('func','in_file')]),
                         (gunzip_func, smooth,       [('out_file','in_files')]),
                         (smooth, modelspec,         [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,    [('session_info','session_info')]),
                         (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),

                         (level1estimate, level1conest, [('spm_mat_file','spm_mat_file'),
                                                         ('beta_images','beta_images'),
                                                         ('residual_image','residual_image')
                                                         ]),
                         (level1conest, datasink, [('spm_mat_file','{}.@spm_mat'.format(ifold)),
                                                   ('spmT_images', '{}.@T'.format(ifold)),
                                                   ('con_images',  '{}.@con'.format(ifold)),
                                                   ('spmF_images', '{}.@F'.format(ifold)),
                                                   ])
                         ])

    # run the 1st analysis
    analysis1st.run('MultiProc',{'n_procs': 80,'memory_gb': 100})
    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")