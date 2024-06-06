import os
import time
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy import ndimage

def keep_largest_blob(data, debug=False):
    blob_map, _ = ndimage.label(data)
    counts = list(np.bincount(blob_map.flatten()))  # number of pixels in each blob
    if len(counts) <= 1: return data  # no foreground
    if debug: print(f"size of second largest blob: {sorted(counts)[-2]}")
    key_second = counts.index(sorted(counts)[-2])
    return (blob_map == key_second).astype(np.uint8)


def keep_largest_blob_multilabel(data, class_map, rois):
    """
    Keep the largest blob for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in tqdm(rois):
        idx = class_map_inv[roi]
        data_roi = data == idx
        cleaned_roi = keep_largest_blob(data_roi) > 0.5
        data[data_roi] = 0   # Clear the original ROI in data
        data[cleaned_roi] = idx   # Write back the cleaned ROI into data
    # print(f"  keep_largest_blob_multilabel took {time.time() - st:.2f}s")
    return data

def remove_outside_of_mask(seg_path, mask_path, addon=1):
    """
    Remove all segmentations outside of mask.

    seg_path: path to nifti file
    mask_path: path to nifti file
    """
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata()
    mask = nib.load(mask_path).get_fdata()
    mask = binary_dilation(mask, iterations=addon)
    seg[mask == 0] = 0
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), seg_img.affine), seg_path)


def main():
    mask_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/code_test_folder/for_evaluation/500epochs/BL_scratch_DA/'
    save_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/code_test_folder/for_evaluation/500epochs/BL_scratch_DA_V2/'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    class_map = {
        1: "Segment_1",
        2: "Segment_2",
        3: "Segment_3"}
    rois = ["Segment_1", "Segment_2", "Segment_3"]
    for file in os.listdir(mask_dir):
        if file.endswith('.nii.gz'):
            print(f'start processing {file}...')
            nib_img = nib.load(mask_dir + file)
            mask = nib_img.get_fdata()
            keep_largest_blob_mask = keep_largest_blob_multilabel(mask, class_map, rois)
            save_path = save_dir + file
            print(f'saving to {save_path}...')
            nib.save(nib.Nifti1Image(keep_largest_blob_mask, nib_img.affine), save_path)


if __name__ == '__main__':
    main()
