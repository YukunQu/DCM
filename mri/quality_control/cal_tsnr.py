#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 21:13:00 2021

@author: quyukun
"""

import os
from nipype.algorithms.confounds import TSNR


def cal_tsnr(infile,outpath):
    tsnr = TSNR()
    tsnr.inputs.in_file = infile
    filename = os.path.basename(infile)
    tsnr.inputs.tsnr_file = outpath+'/tsnr_' + filename
    tsnr.inputs.mean_file = outpath+'/mean_' + filename
    tsnr.inputs.stddev_file = outpath+'/stddev_' + filename
    tsnr.run()
    
    
if __name__ == "__main__":
    prepath = r'/nfs/s2/userhome/quyukun/workingdir/fmriprep/data/bold/derivatives/fmriprep'
    file = os.path.join(prepath,'sub-{}','ses-{}','func','sub-{}_ses-{}_task-{}_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    outpath_prefix = r'/nfs/s2/userhome/quyukun/workingdir/fmriprep/data/bold/tsnr'
    for subid in ['01']:
        for ses in ['1']:
            for task in ['emotion']:
                for run in ['3']:
                    infile = file.format(subid,ses,subid,ses,task,run)
                    print('Infile:',infile)
                    outpath = os.path.join(outpath_prefix,'sub-{}'.format(subid))
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
                    outpath = os.path.join(outpath,'ses-{}'.format(ses))
                    print('Outpath', outpath)
                    if os.path.exists(outpath):
                        cal_tsnr(infile, outpath)
                    else:
                        os.mkdir(outpath)
                        cal_tsnr(infile, outpath)