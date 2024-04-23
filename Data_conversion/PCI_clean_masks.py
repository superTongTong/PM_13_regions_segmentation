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


def remove_small_blobs(img: np.ndarray, interval=[10, 30], debug=False) -> np.ndarray:
    """
    Find blobs/clusters of same label. Remove all blobs which have a size which is outside of the interval.

    Args:
        img: Binary image.
        interval: Boundaries of the sizes to remove.
        debug: Show debug information.
    Returns:
        Detected blobs.
    """
    mask, number_of_blobs = ndimage.label(img)
    if debug: print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    # If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1: return img

    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1  # set everything else to 1

    if debug:
        print(f"counts: {sorted(counts)[::-1]}")
        _, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))

    return mask


def remove_small_blobs_multilabel(data, class_map, rois, interval=[10, 30], debug=False):
    """
    Remove small blobs for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}

    for roi in tqdm(rois):
        idx = class_map_inv[roi]
        data_roi = (data == idx)
        cleaned_roi = remove_small_blobs(data_roi, interval, debug) > 0.5  # Remove small blobs from this ROI
        data[data_roi] = 0  # Clear the original ROI in data
        data[cleaned_roi] = idx  # Write back the cleaned ROI into data

    # print(f"  remove_small_blobs_multilabel took {time.time() - st:.2f}s")
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
    mask_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/masks_v2/s0164.nii.gz'
    class_map = {
        1: "Segment_1",
        2: "Segment_2",
        3: "Segment_3"}
    rois = ["Segment_1", "Segment_2", "Segment_3"]
    nib_img = nib.load(mask_dir)
    mask = nib_img.get_fdata()
    # remove_small_blobs_mask = remove_small_blobs_multilabel(mask, class_map, rois, interval=[10, 30], debug=True)

    keep_largest_blob_mask = keep_largest_blob_multilabel(mask, class_map, rois)

    # save_path1 = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/masks_v2/s0164_cleaned_remove_small.nii.gz'
    # nib.save(nib.Nifti1Image(remove_small_blobs_mask, nib_img.affine), save_path1)
    save_path2 = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/masks_v2/s0164_cleaned_keep_largest.nii.gz'
    nib.save(nib.Nifti1Image(keep_largest_blob_mask, nib_img.affine), save_path2)


if __name__ == '__main__':
    main()
