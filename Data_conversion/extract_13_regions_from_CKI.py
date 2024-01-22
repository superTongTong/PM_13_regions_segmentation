import slicerio
import nrrd
from Data_conversion.nrrd_to_nifiti_conversion import nifti_write
import time
import nibabel as nib
from nn_algorithm import crop_image, divide_background, restore_cropped_image


def extract_3_regions():
    ''''
    following code is to read the segmentation file and print out the segment names and number of segments in the file
    '''

    # seg_40_file_path_1 = '../data/Segmentations_00066.nrrd'
    # seg_13_file_path_2 = 'cki_test_data.nrrd'

    # # Load the segmentation data
    # segmentation_info = slicerio.read_segmentation_info(seg_40_file_path_1)
    #
    # number_of_segments = len(segmentation_info["segments"])
    # print(f"Number of segments: {number_of_segments}")
    #
    # segment_names = slicerio.segment_names(segmentation_info)
    # print(f"Segment names: {', '.join(segment_names)}")

    # segment0 = slicerio.segment_from_name(segmentation_info, segment_names[0])
    # print("First segment info:\n" + json.dumps(segment0, sort_keys=False, indent=4))

    ''''
    following code is to extract the segmentations from the original segmentation file
    '''
    # # Get a list of .nrrd files in a directory
    # nrrd_files = glob.glob('path/to/nrrd/directory/*.nrrd')
    # for file in nrrd_files:
    #     print(file)
    input_filename = '../data/Segmentations_00066.nrrd'
    # load the segmentation data information
    segmentation_info = slicerio.read_segmentation_info(input_filename)
    segment_names = slicerio.segment_names(segmentation_info)

    # create an empty list to store the segment names and labels
    segment_names_to_labels = []

    # create list of name and lables for the 13 regions
    # example format: segment_names_to_labels = [("Segment_1", 1), ("Segment_1_1", 2), ("Segment_1_2", 3)]
    for i in range(1, 4):
        sge_name_label = (segment_names[i], i+1)
        segment_names_to_labels.append(sge_name_label)

    voxels_data, header = nrrd.read(input_filename)

    # extract the 13 regions from the original segmentation file
    extracted_voxels, extracted_header = slicerio.extract_segments(voxels_data, header, segmentation_info,
                                                                   segment_names_to_labels)
    extracted_header['dimension'] = 3
    # nrrd.write('extracted_13_regions_from_raw.nrrd', extracted_voxels, extracted_header)

    # extracted_data = nrrd.read('extracted_13_regions_from_raw.nrrd')
    nifti_write(extracted_voxels, extracted_header, prefix='extracted_3_regions_from_CKI')


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


if __name__ == "__main__":
    start_time = time.time()

    # extract_3_regions()
    nearest_neighbour()
    total_time = time.time() - start_time
    print(f"--- {total_time:.2f} seconds ---")
