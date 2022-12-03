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


def firstLevel_noPhi(subject_list,set_id,runs,ifold,configs):
    glm_type = configs['glm_type']
    if glm_type == 'separate_hexagon_2phases_correct_trials':
        firstLevel_noPhi_separate(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'separate_hexagon_correct_trials_train':
        firstLevel_noPhi_separate(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'separate_hexagon_2phases_all_trials':
        firstLevel_noPhi_separate_all_trials(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'whole_hexagon_correct_trials':
        firstLevel_noPhi_whole_correct_trials(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'whole_hexagon_all_trials':
        firstLevel_noPhi_whole_all_trials(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'alignPhi_separate_correct_trials':
        firstLevel_alignPhi_separate_correct_trials(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'separate_hexagon_difficult':
        firstLevel_noPhi_difficult(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'RSA':
        firstLevel_RSA(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'fir_hexagon':
        firstLevel_noPhi_fir(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'game2_align_game1':
        pass
    elif glm_type == 'm2_hexagon_correct_trials':
        firstLevel_m2_hexagon(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'm2plus_hexagon_correct_trials':
        firstLevel_m2_hexagon(subject_list,set_id,runs,ifold,configs)
    elif glm_type == 'decision_hexagon_correct_trials':
        firstLevel_decision_hexagon(subject_list,set_id,runs,ifold,configs)
    else:
        raise Exception("The glm type-{} is not supported.".format(glm_type))


def run_info_separate(ev_file,motions_file=None):
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
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    if conditions == ['M1','M2_corr','M2_error','decision_corr','decision_error']:
        pmod = [None,run_pmod,None,run_pmod,None]
        orth = ['No','No','No','No','No']
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


def firstLevel_noPhi_separate(subject_list,set_id,runs,ifold,configs):
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
    condition_names = ['M2_corrxcos^1','M2_corrxsin^1','decision_corrxcos^1','decision_corrxsin^1',
                       'M2_corr','decision_corr']

    # contrastst
    cont01 = ['m2_cos',        'T', condition_names,  [1,0,0,0,0,0]]
    cont02 = ['m2_sin',        'T', condition_names,  [0,1,0,0,0,0]]

    cont03 = ['decision_cos',  'T', condition_names,  [0,0,1,0,0,0]]
    cont04 = ['decision_sin',  'T', condition_names,  [0,0,0,1,0,0]]

    cont05 = ['m2_hexagon',       'F', [cont01, cont02]]
    cont06 = ['decision_hexagon', 'F', [cont03, cont04]]

    cont07 = ['m2_corr',        'T', condition_names,  [0,0,0,0,1,0]]
    cont08 = ['decision_corr',  'T', condition_names,  [0,0,0,0,0,1]]

    cont09 =  ['cos',  'T',condition_names,  [1,0,1,0,0,0]]
    cont010 = ['sin',  'T',condition_names,  [0,1,0,1,0,0]]
    cont011 = ['hexagon', 'F', [cont09, cont010]]

    contrast_list = [cont01,cont02,cont03,cont04,cont05,cont06,cont07,cont08,cont09,cont010,cont011]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_separate),
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
    analysis1st.run('MultiProc',{'n_procs': 50,'memory_gb' : 100})
    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")



def run_info_m2(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2_corr','M2_error','decision']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    if conditions == ['M1','M2_corr','M2_error','decision']:
        pmod = [None,run_pmod,None,None]
        orth = ['No','No','No','No']
    elif conditions == ['M1','M2_corr','decision']:
        pmod = [None,run_pmod,None]
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


def firstLevel_m2_hexagon(subject_list,set_id,runs,ifold,configs):
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
    condition_names = ['M2_corrxcos^1','M2_corrxsin^1','M2_corr','decision']

    # contrastst
    cont01 = ['m2_cos',        'T', condition_names,  [1,0,0,0]]
    cont02 = ['m2_sin',        'T', condition_names,  [0,1,0,0]]
    cont03 = ['m2_corr',       'T', condition_names,  [0,0,1,0]]
    cont04 = ['decision',      'T', condition_names,  [0,0,0,1]]
    cont05 = ['m2_hexagon',    'F', [cont01, cont02]]
    contrast_list = [cont01,cont02,cont03,cont04,cont05]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_m2),
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
    analysis1st.run('MultiProc',{'n_procs': 50,'memory_gb': 100})
    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")


def run_info_decision(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2','decision_corr','decision_error']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    if conditions == ['M1','M2','decision_corr','decision_error']:
        pmod = [None,None,run_pmod,None]
        orth = ['No','No','No','No']
    elif conditions == ['M1','M2','decision_corr']:
        pmod = [None,None,run_pmod]
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


def firstLevel_decision_hexagon(subject_list,set_id,runs,ifold,configs):
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
    condition_names = ['decision_corrxcos^1','decision_corrxsin^1','M2','decision_corr']

    # contrastst
    cont01 = ['decision_cos',        'T', condition_names,  [1,0,0,0]]
    cont02 = ['decision_sin',        'T', condition_names,  [0,1,0,0]]
    cont03 = ['M2',                  'T', condition_names,  [0,0,1,0]]
    cont04 = ['decision_corr',       'T', condition_names,  [0,0,0,1]]
    cont05 = ['decision_hexagon',    'F', [cont01, cont02]]
    contrast_list = [cont01,cont02,cont03,cont04,cont05]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_decision),
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
    analysis1st.run('MultiProc',{'n_procs': 50,'memory_gb': 100})
    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")



def run_info_separate_all_trials(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2','decision']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    motions_df = pd.read_csv(motions_file,sep='\t')

    motion_columns = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z',
                      'csf','white_matter']

    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,
                     pmod=[None,run_pmod,run_pmod],
                     orth=['No','No','No'],regressor_names=motion_columns,regressors=motions)
    return run_info


def firstLevel_noPhi_separate_all_trials(subject_list,set_id,runs,ifold,configs):

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

    templates = {'func': pjoin(data_root,'sub-{subj_id}/func',func_name),
                 'event': pjoin(event_dir,task, glm_type, 'sub-{subj_id}', ifold, event_name),
                 'regressors':pjoin(data_root,'sub-{subj_id}/func',regressor_name)
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

    condition_names = ['M2xcos^1','M2xsin^1','decisionxcos^1','decisionxsin^1','M2','decision']

    # contrastst
    cont01 = ['m2_cos',        'T', condition_names,  [1,0,0,0,0,0]]
    cont02 = ['m2_sin',        'T', condition_names,  [0,1,0,0,0,0]]

    cont03 = ['decision_cos',  'T', condition_names,  [0,0,1,0,0,0]]
    cont04 = ['decision_sin',  'T', condition_names,  [0,0,0,1,0,0]]

    cont05 = ['m2_hexagon',       'F', [cont01, cont02]]
    cont06 = ['decision_hexagon', 'F', [cont03, cont04]]

    cont07 = ['m2',             'T', condition_names,  [0,0,0,0,1,0]]
    cont08 = ['decision',       'T', condition_names,  [0,0,0,0,0,1]]

    cont09 =  ['cos',  'T',condition_names,  [1,0,1,0,0,0]]
    cont010 = ['sin',  'T',condition_names,  [0,1,0,1,0,0]]
    cont011 = ['hexagon', 'F', [cont09, cont010]]

    contrast_list = [cont01,cont02,cont03,cont04,cont05,cont06,cont07,cont08,cont09,cont010,cont011]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_separate_all_trials),
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

    mask_img = r'/mnt/workdir/DCM/docs/Reference/Mask/res-02_desc-brain_mask_6mm.nii'
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

                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth, modelspec,        [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,[('session_info','session_info')]),
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
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")



def run_info_separate_3phases_correct_trial(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2_corr','M2_error','decision_corr','decision_error','planning_corr','planning_error',
                 'pressButton']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    motions_df = pd.read_csv(motions_file,sep='\t')

    motion_columns = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']

    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,
                     pmod=[None,run_pmod,None,run_pmod,None,run_pmod,None,None],
                     orth=['No','No','No','No','No','No','No','No'],regressor_names=motion_columns,regressors=motions)
    return run_info


def firstLevel_noPhi_separate_3phases_correct_trial(subject_list, set_id, runs, ifold, configs):

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

    templates = {'func': pjoin(data_root,'sub-{subj_id}/func',func_name),
                 'event': pjoin(event_dir,task, glm_type, 'sub-{subj_id}', ifold, event_name),
                 'regressors':pjoin(data_root,'sub-{subj_id}/func',regressor_name)
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
    condition_names = ['M2_corrxcos^1','M2_corrxsin^1','decision_corrxcos^1','decision_corrxsin^1',
                       'M2_corr','M2_error','decision_corr','decision_error',
                       'planning_corrxcos^1','planning_corrxsin^1']

    # contrastst
    cont01 = ['m2_cos',        'T', condition_names,  [1,0,0,0,0,0,0,0,0,0]]
    cont02 = ['m2_sin',        'T', condition_names,  [0,1,0,0,0,0,0,0,0,0]]

    cont03 = ['decision_cos',  'T', condition_names,  [0,0,1,0,0,0,0,0,0,0]]
    cont04 = ['decision_sin',  'T', condition_names,  [0,0,0,1,0,0,0,0,0,0]]

    cont05 = ['m2_hexagon',       'F', [cont01, cont02]]
    cont06 = ['decision_hexagon', 'F', [cont03, cont04]]

    cont07 = ['m2',             'T', condition_names,  [0,0,0,0,1,1,0,0,0,0]]
    cont08 = ['decision_corr',  'T', condition_names,  [0,0,0,0,0,0,1,0,0,0]]

    cont09 =  ['cos',  'T',condition_names,  [1,0,1,0,0,0,0,0,1,0]]
    cont010 = ['sin',  'T',condition_names,  [0,1,0,1,0,0,0,0,0,1]]
    cont011 = ['hexagon', 'F', [cont09, cont010]]

    cont012 = ['planning_cos', 'T',condition_names, [0,0,0,0,0,0,0,0,1,0]]
    cont013 = ['planning_sin', 'T',condition_names, [0,0,0,0,0,0,0,0,0,1]]
    cont014 = ['planning_hexagon', 'F', [cont012,cont013]]

    cont015 = ['hexagon', 'F', [cont01, cont02,cont03,cont04,cont012,cont013]]

    contrast_list = [cont01,cont02,cont03,cont04,cont05,cont06,cont07,cont08,cont09,cont010,cont011,cont012,
                     cont013,cont014,cont015]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])
    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_separate_3phases_correct_trial),
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

    mask_img = r'/mnt/workdir/DCM/docs/Reference/Mask/res-02_desc-brain_mask_6mm.nii'
    # Level1Design - Generates an SPM design matrix
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0,0]}},
                                     timing_units='secs',
                                     interscan_interval=3.,
                                     model_serial_correlations='AR(1)',
                                     microtime_resolution=49,
                                     microtime_onset=24,
                                     mask_image=mask_img,
                                     flags={'mthresh':'-Inf',
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

                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth, modelspec,        [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,[('session_info','session_info')]),
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
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")

def run_info_difficult(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod1_names = []
    pmod1_params = []
    pmod1_polys = []

    pmod2_names = []
    pmod2_params = []
    pmod2_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','M2_corr','M2_error','decision_corr','decision_error','pressButton']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos','difficult']:
            pmod2_names.append(condition)
            pmod2_params.append(group[1].modulation.tolist())
            pmod2_polys.append(1)

            if condition in ['sin','cos']:
                pmod1_names.append(condition)
                pmod1_params.append(group[1].modulation.tolist())
                pmod1_polys.append(1)

    motions_df = pd.read_csv(motions_file,sep='\t')

    motion_columns = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                      'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                      'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
                      'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
                      'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                      'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

    """motion_columns= ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']"""

    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_pmod1 = Bunch(name=pmod1_names,param=pmod1_params,poly=pmod1_polys)
    run_pmod2 = Bunch(name=pmod2_names,param=pmod2_params,poly=pmod2_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,
                     pmod=[None,run_pmod1,None,run_pmod2,None,None],
                     orth=['No','No','No','No','No','No'],regressor_names=motion_columns,regressors=motions)
    return run_info


def firstLevel_noPhi_difficult(subject_list,set_id,runs,ifold,configs):

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

    templates = {'func': pjoin(data_root,'sub-{subj_id}/func',func_name),
                 'event': pjoin(event_dir,'sub-{subj_id}',task,glm_type, ifold, event_name),
                 'regressors':pjoin(data_root,'sub-{subj_id}/func',regressor_name)
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
    condition_names = ['M2_corrxcos^1','M2_corrxsin^1','decision_corrxcos^1','decision_corrxsin^1',
                       'M2_corr','M2_error','decision_corr','decision_error','decision_corrxdifficult^1']

    # contrastst
    cont01 = ['m2_cos',        'T', condition_names,  [1,0,0,0,0,0,0,0,0]]
    cont02 = ['m2_sin',        'T', condition_names,  [0,1,0,0,0,0,0,0,0]]

    cont03 = ['decision_cos',  'T', condition_names,  [0,0,1,0,0,0,0,0,0]]
    cont04 = ['decision_sin',  'T', condition_names,  [0,0,0,1,0,0,0,0,0]]

    cont05 = ['m2_hexagon',       'F', [cont01, cont02]]
    cont06 = ['decision_hexagon', 'F', [cont03, cont04]]

    cont07 = ['m2',             'T', condition_names,  [0,0,0,0,1,1,0,0,0]]
    cont08 = ['decision_corr',  'T', condition_names,  [0,0,0,0,0,0,1,0,0]]

    cont09 =  ['cos',  'T',condition_names,  [1,0,1,0,0,0,0,0,0]]
    cont010 = ['sin',  'T',condition_names,  [0,1,0,1,0,0,0,0,0]]
    cont011 = ['hexagon', 'F', [cont09, cont010]]

    cont012 = ['decision_corr-error',  'T', condition_names,  [0,0,0,0,0,0,1,-1,0]]
    cont013 = ['difficult', 'T',condition_names,  [0,0,0,0,0,0,0,0,1]]

    contrast_list = [cont01,cont02,cont03,cont04,cont05,cont06,cont07,cont08,cont09,cont010,cont011,cont012,cont013]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_difficult),
                        name='runsinfo',
                        iterfield=['ev_file','motions_file'])

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=tr,
                                     high_pass_filter_cutoff=128.,
                                     ),
                     name='modelspec')

    # Level1Design - Generates an SPM design matrix
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0,0]}},
                                     timing_units='secs',
                                     interscan_interval=3.,
                                     model_serial_correlations='AR(1)',
                                     microtime_resolution=49,
                                     microtime_onset=24,
                                     flags={'mthresh':float('-inf')}),
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

                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth, modelspec,        [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,[('session_info','session_info')]),
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
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")


def run_info_whole_correct_trials(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations  = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','infer_corr','infer_error']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    if conditions == ['M1','infer_corr','infer_error']:
        pmod = [None,run_pmod,None]
        orth = ['No','No','No']
    elif conditions == ['M1','infer_corr']:
        pmod = [None,run_pmod]
        orth = ['No','No']
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


def firstLevel_noPhi_whole_correct_trials(subject_list,set_id,runs,ifold,configs):
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
    condition_names = ['infer_corrxcos^1','infer_corrxsin^1','M1']

    # contrasts
    cont01 = ['infer_corrxcos^1',   'T', condition_names, [1, 0, 0]]
    cont02 = ['infer_corrxsin^1',   'T', condition_names, [0, 1, 0]]

    cont03 = ['hexagon', 'F', [cont01, cont02]]

    cont04 = ['visual','T',condition_names, [0, 0, 1]]

    contrast_list = [cont01, cont02, cont03, cont04]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_whole_correct_trials),
                        name='runsinfo',
                        iterfield = ['ev_file','motions_file'])

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=tr,
                                     high_pass_filter_cutoff=100.,
                                     ),
                     name='modelspec')

    mask_img = r'/mnt/workdir/DCM/docs/Mask/res-02_desc-brain_mask.nii'
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

                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth, modelspec,        [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,[('session_info','session_info')]),
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
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")


def run_info_whole_all_trials(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['M1','inference']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append(group[1].duration.tolist())
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    motions_df = pd.read_csv(motions_file,sep='\t')

    motion_columns = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,pmod=[None,run_pmod],
                     orth=['No','No'],
                     regressor_names=motion_columns,regressors=motions)
    return run_info


def firstLevel_noPhi_whole_all_trials(subject_list,set_id,runs,ifold,configs):
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

    templates = {'func': pjoin(data_root,'sub-{subj_id}/func',func_name),
                 'event': pjoin(event_dir,task, glm_type,'sub-{subj_id}',ifold, event_name),
                 'regressors':pjoin(data_root,'sub-{subj_id}/func',regressor_name)
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
    condition_names = ['inferencexcos^1','inferencexsin^1','M1']

    # contrasts
    cont01 = ['inferencexcos^1',   'T', condition_names, [1, 0, 0]]
    cont02 = ['inferencexsin^1',   'T', condition_names, [0, 1, 0]]

    cont03 = ['hexagon', 'F', [cont01, cont02]]
    cont04 = ['M1',      'T', condition_names, [0, 0, 1]]
    contrast_list = [cont01, cont02, cont03, cont04]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_whole_all_trials),
                        name='runsinfo',
                        iterfield = ['ev_file','motions_file'])

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=tr,
                                     high_pass_filter_cutoff=100.,
                                     ),
                     name='modelspec')

    mask_img = r'/mnt/workdir/DCM/docs/Reference/Mask/res-02_desc-brain_mask_6mm.nii'
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

                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth, modelspec,        [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,[('session_info','session_info')]),
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
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")


def run_info_fir(ev_file,dur=2,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    trial_con = ['infer_corr', 'infer_error']
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        if condition in trial_con:
            conditions.append(condition)
            onsets.append(group[1].onset.tolist())
            durations.append([10]*len(group[1].onset.tolist()))
        elif condition in ['sin','cos']:
            pmod_names.append(condition)
            pmod_params.append(group[1].modulation.tolist())
            pmod_polys.append(1)

    motions_df = pd.read_csv(motions_file,sep='\t')
    motion_columns = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z',
                      'csf','white_matter']
    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_pmod = Bunch(name=pmod_names,param=pmod_params,poly=pmod_polys)
    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,
                     pmod=[run_pmod,None],
                     orth=['No','No'],regressor_names=motion_columns,regressors=motions)
    return run_info


def firstLevel_noPhi_fir(subject_list,set_id,runs,ifold,configs):

    # start cue
    start_time = time.time()
    print("Training set",set_id," ",ifold," start!")
    # set parameters and specify which SPM to use
    tr = 3.
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # Specify input & output stream
    infosource = Node(IdentityInterface(fields=['subj_id']), name="infosource")
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
    condition_names = ['infer_corrxcos^1','infer_corrxsin^1']

    # contrastst
    cont01 = ['cos',     'T', condition_names,  [1,0]]
    cont02 = ['sin',     'T', condition_names,  [0,1]]
    cont03 = ['hexagon', 'F', [cont01, cont02]]

    contrast_list = [cont01,cont02,cont03]

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    smooth = Node(Smooth(fwhm=[8.,8.,8.]), name="smooth")

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file','motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_fir),
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
    # Level1Design - Generates an SPM design matrix
    level1design = Node(Level1Design(bases={'fir': {'length': 10,
                                                    'order': 5},},
                                     timing_units='secs',
                                     interscan_interval=3.,
                                     model_serial_correlations='AR(1)',
                                     microtime_resolution=49,
                                     microtime_onset=25,
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

                         (selectfiles, gunzip_func, [('func','in_file')]),
                         (gunzip_func, smooth,      [('out_file','in_files')]),
                         (smooth, modelspec,        [('smoothed_files','functional_runs')]),

                         (modelspec,level1design,   [('session_info','session_info')]),
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
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")


def firstLevel_RSA(subject_list,set_id,runs,ifold,configs):

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

    templates = {'func': pjoin(data_root,'sub-{subj_id}/func',func_name),
                 'event': pjoin(event_dir,'sub-{subj_id}',task,glm_type, ifold, event_name),
                 'regressors':pjoin(data_root,'sub-{subj_id}/func',regressor_name)
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
    # contrastst
    contrast_list = []
    condition_names = ['pos'+str(i) for i in range(2,25)]
    for index,pos_id in enumerate(range(2, 25)):
        contrast_vector = [0] * 23
        contrast_vector[index] = 1
        contrast_list.append(['pos'+str(pos_id), 'T', condition_names, contrast_vector])

    # Specify Nodes
    gunzip_func = MapNode(Gunzip(), name='gunzip_func',iterfield=['in_file'])

    # prepare event file
    runs_prep = MapNode(Function(input_names=['ev_file', 'motions_file'],
                                 output_names=['run_info'],
                                 function=run_info_RSA),
                        name='runsinfo',
                        iterfield=['ev_file','motions_file'])

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=tr,
                                     high_pass_filter_cutoff=128.,
                                     ),
                     name='modelspec')

    # Level1Design - Generates an SPM design matrix
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0,0]}},
                                     timing_units='secs',
                                     interscan_interval=3.,
                                     model_serial_correlations='AR(1)',
                                     microtime_resolution=49,
                                     microtime_onset=24,
                                     flags={'mthresh':float('-inf')}),
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

                         (selectfiles, gunzip_func, [('func', 'in_file')]),
                         (gunzip_func, modelspec,   [('out_file', 'functional_runs')]),

                         (modelspec,level1design,[('session_info','session_info')]),
                         (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),

                         (level1estimate, level1conest, [('spm_mat_file','spm_mat_file'),
                                                         ('beta_images','beta_images'),
                                                         ('residual_image','residual_image')
                                                         ]),
                         (level1conest, datasink, [('spm_mat_file', '6fold.@spm_mat'),
                                                   ('spmT_images',  '6fold.@T'),
                                                   ('con_images',   '6fold.@con')
                                                   ])
                         ])

    # run the 1st analysis
    analysis1st.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time)/60/60, 2)
    print(f"Run time cost {run_time}")


def run_info_RSA(ev_file,motions_file=None):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    onsets = []
    conditions = []
    durations = []

    ev_info = pd.read_csv(ev_file, sep='\t')
    for group in ev_info.groupby('trial_type'):
        condition = group[0]
        conditions.append(condition)
        onsets.append(group[1].onset.tolist())
        durations.append(group[1].duration.tolist())

    motions_df = pd.read_csv(motions_file,sep='\t')

    motion_columns = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                      'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                      'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
                      'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
                      'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                      'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

    motions = motions_df[motion_columns]
    motions = motions.fillna(0.0).values.T.tolist()

    run_info = Bunch(conditions=conditions,onsets=onsets,durations=durations,
                     pmod=[None,None,None,None,None,None],
                     orth=['No','No','No','No','No','No'],regressor_names=motion_columns,regressors=motions)
    return run_info


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


def firstLevel_alignPhi_separate_correct_trials(subject_list,set_id,runs,ifold,configs):
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

    cont03 = ['alignPhi',               'T', condition_names,  [1,1,0,0]]
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