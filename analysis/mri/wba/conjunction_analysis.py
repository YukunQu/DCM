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
    map1 = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/M2_Decision/Setall/group/covariates/2ndLevel/_contrast_id_ZF_0005/spmT_0002.nii'
    map2 = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/M2_Decision/Setall/group/covariates/2ndLevel/_contrast_id_ZF_0005/spmT_0003.nii'
    outpath = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/M2_Decision/Setall/group/covariates/2ndLevel/_contrast_id_ZF_0005/conjun_age_acc.nii'
    conj_stat_maps(map1,map2,outpath,'threshold',thr=1.31)


#%%
"""
Left-tailed test
Suppose we want to find the p-value associated with a t-score of -0.77 and df = 15 in a left-tailed hypothesis test.

import scipy.stats

#find p-value
scipy.stats.t.sf(abs(-.77), df=15)

0.2266283049085413
The p-value is 0.2266. If we use a significance level of α = 0.05, we would fail to reject the null hypothesis of our hypothesis test because this p-value is not less than 0.05.

Right-tailed test
Suppose we want to find the p-value associated with a t-score of 1.87 and df = 24 in a right-tailed hypothesis test.

import scipy.stats

#find p-value
scipy.stats.t.sf(abs(1.87), df=24)

0.036865328383323424
The p-value is 0.0368. If we use a significance level of α = 0.05, we would reject the null hypothesis of our hypothesis test because this p-value is less than 0.05.

Two-tailed test
Suppose we want to find the p-value associated with a t-score of 1.24 and df = 22 in a two-tailed hypothesis test.

import scipy.stats

#find p-value for two-tailed test
scipy.stats.t.sf(abs(1.24), df=22)*2
0.22803901531680093
"""