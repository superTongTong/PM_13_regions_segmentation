from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree


'''
加速算法思路1： 
先放大13个区域，再筛选出重叠的部分，最后NN分割重叠的区域

思路2：
对label区域填充，在相互删减，可以得出空白区域。
'''

def divide_background(input_data, distance_threshold):

    # Find the coordinates of the background voxels
    bg_coords = np.argwhere(input_data == 0)

    # If there are no background voxels, return the original data
    if len(bg_coords) == 0:
        return input_data

    # Flatten the non-background coordinates for building the k-d tree
    non_bg_coords = np.argwhere(input_data != 0)

    # Build a k-d tree for efficient nearest neighbor searches
    kdtree = cKDTree(non_bg_coords)

    # Create a new 3D array to store the divided background
    new_data = np.copy(input_data)

    # Iterate through each background voxel
    for bg_coord in bg_coords:
        # Query the k-d tree to find the nearest neighbor in the non-background coordinates
        _, closest_non_bg_index = kdtree.query(bg_coord)

        # Retrieve the non-background index
        closest_non_bg_coord = non_bg_coords[closest_non_bg_index]

        # Get the distance to the nearest neighbor
        distance = np.linalg.norm(bg_coord - closest_non_bg_coord)

        # Assign the background voxel to the value of its closest non-background neighbor
        if distance <= distance_threshold:
            new_data[tuple(bg_coord)] = input_data[tuple(closest_non_bg_coord)]

    return new_data


def check_data_info():

    lable_path = "./masks/13_regions/knn_part.nii.gz"
    lable = Path(lable_path)
    lable_data = nib.load(lable)
    lable_in_shape = lable_data.shape
    lable_in_zooms = lable_data.header.get_zooms()

    print("lable shape :", lable_in_shape)
    print("lable zooms :", lable_in_zooms)


def find_overlap_and_create_seg_for_knn(segmentations):

    num_segmentations = len(segmentations)
    overlap_layers = np.zeros_like(segmentations[0])
    seg_for_knn = np.zeros_like(segmentations[0])
    # Iterate through pairs of segmentations
    for i in range(num_segmentations - 1):
        for j in range(i + 1, num_segmentations):

            seg1 = segmentations[i]
            seg2 = segmentations[j]
            # Create a binary mask indicating the overlapped part
            overlap_mask = ((seg1 > 0.5) & (seg2 > 0.5)).astype(int)

            # Update the overlap layer
            overlap_layers[overlap_mask == 1] = 1

    # Create a binary mask contains all the regions and set the overlapped part to 0
    for count, data in enumerate(segmentations):
        seg_for_knn[data > 0.5] = count + 1
    seg_for_knn[overlap_layers == 1] = 0

    return overlap_layers, seg_for_knn


def crop_image(input_array):
    # Find the indices of nonzero values
    nonzero_indices = np.argwhere(input_array)

    if len(nonzero_indices) == 0:
        # No nonzero values found
        print("No nonzero values found in the input array")
        return None

    # Extract the minimum and maximum indices along each axis
    min_indices = np.min(nonzero_indices, axis=0)
    max_indices = np.max(nonzero_indices, axis=0)

    # Create the sub-array based on the boundary defined by nonzero values
    sub_array = input_array[min_indices[0]:max_indices[0],
                            min_indices[1]:max_indices[1],
                            min_indices[2]:max_indices[2]]

    return sub_array, max_indices, min_indices


def restore_cropped_image(sub_array, start_indices, end_indices, original_shape=(512, 512, 214)):
    # Create an array of zeros with the original shape
    result_array = np.zeros(original_shape)

    if sub_array is not None:

        # Copy the values from the sub-array to the corresponding positions in the result array
        result_array[start_indices[0]:end_indices[0],
                     start_indices[1]:end_indices[1],
                     start_indices[2]:end_indices[2]] = sub_array

    return result_array


def main():
    # Load the NIfTI files
    source_file_path = "./masks/13_regions/for_knn_test.nii.gz"
    segs = nib.load(source_file_path)

    # Convert to a list according to the last dimension
    seg_len = segs.shape[-1]
    list_of_seg = np.split(segs.get_fdata(), seg_len, axis=-1)
    list_of_seg_reshape = [arr.squeeze() for arr in list_of_seg]
    overlap_data, data_for_knn = find_overlap_and_create_seg_for_knn(list_of_seg_reshape)

    # source_file_path = "./masks/13_regions/knn_part.nii.gz"
    # data_for_knn = nib.load(source_file_path)
    fdata_knn = data_for_knn#.get_fdata()
    # crop the image
    cropped_data, start_point, end_point = crop_image(fdata_knn)

    # Divide the background
    after_nn_data = divide_background(cropped_data, 3)

    # Add zeros to the sub-array to restore it to the original shape
    original_shape = fdata_knn.shape
    restored_shape = restore_cropped_image(after_nn_data, end_point, start_point, original_shape)

    # Save the new NIfTI file
    knn_nifti = nib.Nifti1Image(restored_shape, data_for_knn.affine)
    nib.save(knn_nifti, "./masks/13_regions/after_knn_processed.nii.gz")


if __name__ == "__main__":
    main()

    # check_data_info()





