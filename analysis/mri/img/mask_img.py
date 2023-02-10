from nilearn.image import load_img,new_img_like


def mask_img(source_img,mask,out_img):
    source_img = load_img(source_img)
    mask = load_img(mask)

    source_data = source_img.get_fdata()
    mask_data = mask.get_fdata()
    source_data[mask_data == 0] = 0
    masked_img = new_img_like(source_img, source_data)
    masked_img.to_filename(out_img)