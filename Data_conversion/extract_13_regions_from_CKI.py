import slicerio
import nrrd
from Data_conversion.nrrd_to_nifiti_conversion import nifti_write
import time
import nibabel as nib
from nn_algorithm_under_develop import crop_image, divide_background, restore_cropped_image
import os
import glob
from tqdm import tqdm
import dicom2nifti
from resample_data_itk import resample_img
import SimpleITK as sitk


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


def extract_3_regions(input_folder, output_folder):

    ''''
    following code is to extract the segmentations from the original segmentation file
    '''
    # Get a list of .nrrd files in a directory
    nrrd_files = glob.glob(input_folder)
    count = 1
    for file in tqdm(nrrd_files):

        seg_label = f's{file[-13:-9]}'
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
        # directory = '../data/three_regions_segmentation_orig/masks_1_3'
        os.makedirs(output_folder, exist_ok=True)
        save_dir = os.path.join(output_folder, seg_label)

        nifti_write(extracted_voxels, extracted_header, prefix=save_dir)
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


def start_dicom_to_nifti(input_folder, output_folder):
    count = 0
    for root, dirs, files in os.walk(input_folder):
        # Create the corresponding directory structure in the output folder
        # relative_path = os.path.relpath(root, input_folder)
        # output_path = os.path.join(output_folder, relative_path)
        # os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.endswith(".dcm"):
                name = f"s{file[1:5]}_0000"
                converted_data_save_dir = os.path.join(output_folder, name)
                dcm_to_nifti(root, converted_data_save_dir)
                count += 1
                print(f"file {name} have been processed.{count}")
                break  # Add the folder once and move on to the next


def dcm_to_nifti(input_path, output_path):
    """
    Uses dicom2nifti package (also works on windows)

    input_path: a directory of dicom slices
    output_path: a nifti file path
    """
    dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=False)


def resample_image_from_folder(data_path, output_path, is_label=False):

    # Get all nii.gz files in data_path
    file_names = os.listdir(data_path)
    # file_names = [f for f in file_names if '.nii.gz' in f]
    # file_names.sort()
    for f in tqdm(file_names):
        # Read the .nii.gz file
        itk_image = sitk.ReadImage(os.path.join(data_path, f))

        # Resample the image for label!!! set to True
        # resampled_sitk_img = resample_img(itk_image, is_label=True)

        # Resample the image for image!!! set to false
        resampled_sitk_img = resample_img(itk_image, is_label=is_label)

        # Write the resampled image
        sitk.WriteImage(resampled_sitk_img, os.path.join(output_path, f))


if __name__ == "__main__":
    start_time = time.time()

    # note: step 1 and 2 can be skipped since for 3/13 regions seg use the same 60 scans,
    # and they are already converted to nifti.gz format, they are located at
    # ../data/three_regions_segmentation_orig/First_60_cases

    # 'step 1: convert the image from dicom files to nifti files'
    # raw_image_folder = '../code_test_folder/test_image'
    #
    # image_nifti_folder = tmp_dir / 'test_image_nifti'
    # start_dicom_to_nifti(raw_image_folder, image_nifti_folder)
    #
    # 'step 2: the converted nifti files are .nii format, we need to compress them to .nii.gz format'
    # path_for_compressed_data = tmp_dir / 'test_image_nifti_gz'
    # compress_raw_image(image_nifti_folder, path_for_compressed_data)
    '''replace the input and output path for step 3,4,5 with the correct path'''
    'step 3: extract the 3 regions from the original segmentation file'

    raw_label_folder = '../code_test_folder/test_mask/*.nrrd'
    mask_save_dir = '../code_test_folder/extract_3_regions_orig'

    extract_3_regions(raw_label_folder, mask_save_dir)

    'step 4: resample the image spacing to 1.5mm x 1.5mm x 1.5mm'

    raw_image_folder = '../code_test_folder/test_image'
    resampled_image_save_dir = '../code_test_folder/test_image_resampled'
    os.makedirs(resampled_image_save_dir, exist_ok=True)
    resample_image_from_folder(raw_image_folder, resampled_image_save_dir, is_label=False)

    'step 5: resample the label spacing to 1.5mm x 1.5mm x 1.5mm'
    raw_label_folder = '../code_test_folder/extract_3_regions_orig'
    resampled_label_save_dir = '../code_test_folder/extract_3_regions_resampled'
    os.makedirs(resampled_label_save_dir, exist_ok=True)
    resample_image_from_folder(raw_label_folder, resampled_label_save_dir, is_label=True)

    # nearest_neighbour()
    total_time = time.time() - start_time
    print(f"--- {total_time:.2f} seconds ---")
