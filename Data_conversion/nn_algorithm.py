
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import SimpleITK as sitk
import os

def divide_overlapped_region(input_data, orig_data, distance_threshold):

    # Find the coordinates of the background voxels
    ov_coords = np.argwhere(input_data == 20)

    # If there are no background voxels, return the original data
    if len(ov_coords) == 0:
        return input_data

    # Flatten the non-background coordinates for building the k-d tree
    non_bg_coords = np.argwhere(orig_data != 0)

    # Build a k-d tree for efficient nearest neighbor searches
    kdtree = cKDTree(non_bg_coords)

    # Create a new 3D array to store the divided background
    new_data = np.copy(orig_data)

    # Iterate through each background voxel
    for ov_coord in ov_coords:
        # Query the k-d tree to find the nearest neighbor in the non-background coordinates
        _, closest_non_bg_index = kdtree.query(ov_coord)

        # Retrieve the non-background index
        closest_non_bg_coord = non_bg_coords[closest_non_bg_index]

        # Get the distance to the nearest neighbor
        distance = np.linalg.norm(ov_coord - closest_non_bg_coord)

        # Assign the background voxel to the value of its closest non-background neighbor
        if distance <= distance_threshold:
            # new_data[tuple(ov_coord)] = input_data[tuple(closest_non_bg_coord)]
            new_data[tuple(ov_coord)] = orig_data[tuple(closest_non_bg_coord)]

    return new_data


def check_data_info():

    lable_path = "./masks/13_regions/knn_part.nii.gz"
    lable = Path(lable_path)
    lable_data = nib.load(lable)
    lable_in_shape = lable_data.shape
    lable_in_zooms = lable_data.header.get_zooms()

    print("lable shape :", lable_in_shape)
    print("lable zooms :", lable_in_zooms)


def find_non_overlap(closed_image, threshold_image):

    # convert the sitk image to numpy array
    closed_image = sitk.GetArrayFromImage(closed_image)
    threshold_image = sitk.GetArrayFromImage(threshold_image)

    overlap_layers = np.zeros_like(closed_image)
    # seg_for_knn = np.zeros_like(segmentations[0])

    # Create a binary mask indicating the overlapped part
    overlap_mask = ((closed_image > 0.5) & (threshold_image > 0.5)).astype(int)

    # Update the overlap layer, set the overlapped region to 20
    differ_mask = closed_image - overlap_mask
    overlap_layers[differ_mask == 1] = 20

    return overlap_layers


def closing_image(img, kernel_radius):

    # Binary threshold to set all labels to the new label value
    binary_threshold_filter = sitk.BinaryThresholdImageFilter()
    binary_threshold_filter.SetLowerThreshold(1)
    '''
    the upper threshold set 3 for 3 regions, 13 for 13 regions
    '''
    binary_threshold_filter.SetUpperThreshold(13)

    binary_threshold_filter.SetOutsideValue(0)
    binary_threshold_filter.SetInsideValue(1)

    threshold_image = binary_threshold_filter.Execute(img)

    #Apply closing Filter
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()

    closing_filter.SetKernelRadius(kernel_radius)
    closing_filter.SetForegroundValue(1)
    output_image = closing_filter.Execute(threshold_image)

    return output_image, threshold_image


def main():
    # Load the NIfTI files
    source_file_path = "./nn_algo/gt/s0059.nii.gz"
    sitk_orig = sitk.ReadImage(source_file_path, sitk.sitkInt8)
    array_orig = sitk.GetArrayFromImage(sitk_orig)
    c_image, t_image = closing_image(sitk_orig, kernel_radius=[10, 10, 10])
    non_overlapped_voxel = find_non_overlap(c_image, t_image)

    processed_image = divide_overlapped_region(non_overlapped_voxel, array_orig, 10)

    # covert the processed image to sitk image
    img_for_save = sitk.GetImageFromArray(processed_image)
    img_for_save.CopyInformation(sitk_orig)
    save_dir = "./nn_algo/after_nn"
    os.makedirs(save_dir, exist_ok=True)

    # Export Image
    sitk.WriteImage(img_for_save, f"{save_dir}/s0059_closing_v6.nii.gz")


if __name__ == "__main__":
    main()

    # check_data_info()










