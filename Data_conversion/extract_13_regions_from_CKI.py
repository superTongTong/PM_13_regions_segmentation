import slicerio
import nrrd
from Data_conversion.nrrd_to_nifiti_conversion import nifti_write
import time
import nibabel as nib
from nn_algorithm_under_develop import crop_image, divide_background, restore_cropped_image
import os
import glob
from tqdm import tqdm
import gzip

def check_data(input_path):

    ''''
    following code is to read the segmentation file and print out the segment names and number of segments in the file
    '''

    # # Load the segmentation data
    segmentation_info = slicerio.read_segmentation_info(input_path)

    number_of_segments = len(segmentation_info["segments"])
    print(f"Number of segments: {number_of_segments}")

    segment_names = slicerio.segment_names(segmentation_info)
    print(f"Segment names: {', '.join(segment_names)}")

    # segment0 = slicerio.segment_from_name(segmentation_info, segment_names[0])
    # print("First segment info:\n" + json.dumps(segment0, sort_keys=False, indent=4))


def extract_3_regions(input_folder):

    ''''
    following code is to extract the segmentations from the original segmentation file
    '''
    # Get a list of .nrrd files in a directory
    nrrd_files = glob.glob(input_folder)
    count = 1
    for file in nrrd_files:

        seg_label = file[-14:-9]
        print(f'Start processing scan {seg_label}....')
        # input_filename = os.path.join(input_path, file)
        # load the segmentation data information
        segmentation_info = slicerio.read_segmentation_info(file)
        segment_names = slicerio.segment_names(segmentation_info)

        # create an empty list to store the segment names and labels
        segment_names_to_labels = []

        # create list of name and labels for the 13 regions
        # example format: segment_names_to_labels = [("Segment_1", 1), ("Segment_1_1", 2), ("Segment_1_2", 3)]
        for i in range(1, 4): # Currently we onty have 3 regions
            sge_name_label = (segment_names[i], i)
            segment_names_to_labels.append(sge_name_label)

        voxels_data, header = nrrd.read(file)

        # extract the 13 regions from the original segmentation file
        extracted_voxels, extracted_header = slicerio.extract_segments(voxels_data, header, segmentation_info,
                                                                       segment_names_to_labels)
        extracted_header['dimension'] = 3
        directory = '../data/three_regions_segmentation_orig/masks_1_3'
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, seg_label)

        nifti_write(extracted_voxels, extracted_header, prefix=output_path)
        print(f'Finish processing scan {seg_label}, currently {count} files processed.')
        count += 1


def nearest_neighbour():
    source_file_path = "../code_test_folder/extracted_13_regions_from_CKI.nii.gz"
    data_for_knn = nib.load(source_file_path)
    fdata_knn = data_for_knn.get_fdata()
    # crop the image
    cropped_data, start_point, end_point = crop_image(fdata_knn)

    # Divide the background
    after_nn_data = divide_background(cropped_data, 2)

    # Add zeros to the sub-array to restore it to the original shape
    original_shape = fdata_knn.shape
    restored_shape = restore_cropped_image(after_nn_data, end_point, start_point, original_shape)

    # Save the new NIfTI file
    knn_nifti = nib.Nifti1Image(restored_shape, data_for_knn.affine)
    nib.save(knn_nifti, "after_knn_processed.nii.gz")


def compress_raw_image(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    nii_files = os.listdir(input_folder)

    for file in tqdm(nii_files):
        if file.endswith('.nii'):
            # Construct the input and output paths
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file + '.gz')

            # Read the NIfTI file
            nii_img = nib.load(input_path)

            # Save it in .nii.gz format
            nib.save(nii_img, output_path)


if __name__ == "__main__":
    start_time = time.time()
    # data_path = '../data/three_regions_segmentation_orig/region_1_3_22012024/Segmentations_00057.seg.nrrd'
    # input_path = '../data/three_regions_segmentation_orig/region_1_3_22012024/*.nrrd'

    input_folder = '../data/three_regions_segmentation_orig/PM_scans_first60_1mm_nifti'
    output_folder = '../data/three_regions_segmentation_orig/First_60_cases'
    compress_raw_image(input_folder, output_folder)

    # check_data(data_path)
    # extract_3_regions(input_path)
    # nearest_neighbour()
    total_time = time.time() - start_time
    print(f"--- {total_time:.2f} seconds ---")
