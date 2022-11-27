def level2nd_noPhi(subject_list,sub_type,task,glm_type,set_id,contrast_1st):
    spm.SPMCommand().set_mlab_paths(paths='/usr/local/MATLAB/R2020b/toolbox/spm12/')

    # data input and ouput
    infosource = Node(IdentityInterface(fields=['contrast_id', 'subj_id']), name="infosource")
    infosource.iterables = [('contrast_id', contrast_1st)]
    infosource.inputs.subj_id = subject_list

    # SelectFiles
    data_root = '/mnt/workdir/DCM/BIDS/derivatives/Nipype'
    templates = {'cons': pjoin(data_root, f'{task}/{glm_type}/{set_id}/6fold','sub-{subj_id}', '{contrast_id}.nii')}

    # Create SelectFiles node
    selectfiles = MapNode(SelectFiles(templates, base_directory=data_root, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])

    # Initiate DataSink node here
    container_path = f'{task}/{glm_type}/{set_id}/group/{sub_type}'
    datasink = Node(DataSink(base_directory=data_root,container=container_path),
                    name="datasink")

    # Use the following substitutions for the DataSink output
    substitutions = [('_cont_id_', 'con_')]
    datasink.inputs.substitutions = substitutions

    # Node initialize
    onesamplettestdes = Node(OneSampleTTestDesign(), name="onesampttestdes")

    level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),name="level2estimate")

    level2conestimate = Node(EstimateContrast(group_contrast=True),name="level2conestimate")
    # specify contrast
    cont01 = ['Group', 'T', ['mean'], [1]]
    level2conestimate.inputs.contrasts = [cont01]

    level2thresh = Node(Threshold(contrast_index=1,
                                  use_topo_fdr=True,
                                  use_fwe_correction=False,
                                  extent_threshold=0,
                                  height_threshold=0.01,
                                  height_threshold_type='p-value',
                                  extent_fdr_p_threshold=0.05,
                                  ),
                        name="level2thresh")

    # 2nd workflow
    analysis2nd = Workflow(name='work_2nd',base_dir='/mnt/workdir/DCM/BIDS/derivatives/Nipype/working_dir/'
                                                    '{}/{}/{}/group/{}'.format(task,glm_type,set_id,sub_type))
    analysis2nd.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                                    ('subj_id', 'subj_id')]),
                         (selectfiles, onesamplettestdes, [('cons', 'in_files')]),

                         (onesamplettestdes, level2estimate, [('spm_mat_file','spm_mat_file')]),

                         (level2estimate, level2conestimate, [('spm_mat_file','spm_mat_file'),
                                                              ('beta_images','beta_images'),
                                                              ('residual_image','residual_image')]),

                         (level2conestimate, level2thresh, [('spm_mat_file','spm_mat_file'),
                                                            ('spmT_images','stat_image'),]),

                         (level2conestimate, datasink, [('spm_mat_file','2ndLevel.@spm_mat'),
                                                        ('spmT_images', '2ndLevel.@T'),
                                                        ('con_images',  '2ndLevel.@con')]),

                         (level2thresh, datasink,   [('thresholded_map', '2ndLevel.@threshold')])
                         ])
    # run 2nd analysis
    analysis2nd.run('MultiProc', plugin_args={'n_procs': 30})
