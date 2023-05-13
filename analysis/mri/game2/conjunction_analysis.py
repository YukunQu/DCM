from nilearn.image import binarize_img,math_img,get_data,new_img_like
from scipy.stats import norm

def conj_stat_maps(map1,map2,out_file,thr=2.3):
    """
    Runs a conjunction on stat maps
    Args:
        map1 (str): filename of the first stat map in the conjunction
        map2 (str): filename of the second stat map in the conjunction

    Optional:
        out_file (str): output filename.

    Returns:
        out_file (str): output filename (absolute path)
    """

    assert isinstance(thr,float)
    map1_data = get_data(map1)
    map2_data = get_data(map2)

    map1_data[map1_data<=thr] = 0
    map2_data[map2_data<=thr] = 0

    map1_data[map2_data==0] = 0
    map1_data[map1_data>0] =1
    conj_zmap = new_img_like(map1,map1_data)
    conj_zmap.to_filename(out_file)
    return conj_zmap


if __name__ == "__main__":
    map1 = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/distance_spct/Setall/6fold/group_193/mean/M2_corrxdistance_zmap.nii.gz'
    map2 = r'/mnt/workdir/DCM/BIDS/derivatives/Nilearn/game2/value_spct/Setall/6fold/group_193/mean/value_zmap.nii.gz'
    outpath = r'//mnt/workdir/DCM/Result/MRI/Conjunction_analysis/distance_value/conj_game2_distance_value_zmap.nii.gz'
    thr = norm.ppf(1 - 0.001)
    conj_stat_maps(map1,map2,outpath,thr)