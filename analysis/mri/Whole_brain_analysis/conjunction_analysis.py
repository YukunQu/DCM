def conj_stat_maps(map1, map2, out_file=None,method='rigid',thr=1.69):
    """
    Runs a conjunction on stat maps (returning a map with the min t-stat)

    Creates a conjunction of two statistical maps (typically the result of
    EstimateContrast or Threshold from the spm interface).

    Args:
        map1 (str): filename of the first stat map in the conjunction
        map2 (str): filename of the second stat map in the conjunction

    Optional:
        out_file (str): output filename. If None (default), creates
            'conj_map.nii' in current directory

    Returns:
        out_file (str): output filename (absolute path)

    """

    # Imports all required packages so the func can be run as a nipype
    # Function Node
    import nilearn.image as nli
    import os.path

    if (out_file is None):
        out_file = 'conj_map.nii'

    if method == 'rigid':
        conj_tmap = nli.math_img('np.minimum(img1*(img1>0), img2*(img2>0)) + ' + \
                                 'np.maximum(img1*(img1<0), img2*(img2<0))',
                                 img1=map1,
                                 img2=map2)
    elif method == 'threshold':
        assert isinstance(thr,float)
        conj_tmap = nli.math_img('(img1-{})+ (img2-{})'.format(thr,thr),
                                 img1=map1,
                                 img2=map2)

    conj_tmap.to_filename(out_file)
    return os.path.abspath(out_file)

if __name__ == "__main__":
    map1 = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/training_set/trainsetall/group/covariates/2ndLevel/_contrast_id_ZF_0004/spmT_0002.nii'
    map2 = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/training_set/trainsetall/group/covariates/2ndLevel/_contrast_id_ZF_0004/spmT_0003.nii'
    outpath = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/hexagon/specificTo6/training_set/trainsetall/group/covariates/2ndLevel/_contrast_id_ZF_0004/conjun_age_acc.nii'
    conj_stat_maps(map1,map2,outpath,'threshold',thr=1.31)