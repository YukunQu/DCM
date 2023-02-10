from nipype.interfaces.fsl import maths

in_file = r'/mnt/workdir/DCM/docs/Mask/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
out_file = r'/mnt/workdir/DCM/docs/Mask/res-02_desc-brain_mask.nii'

di = maths.DilateImage(in_file=in_file,
                       operation="mean",
                       kernel_shape="sphere",
                       kernel_size=2,
                       out_file=out_file)
di.run()
