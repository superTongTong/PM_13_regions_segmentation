
from tqdm import tqdm
import os
import dicom2nifti
from dicom2nifti.common import validate_slice_increment
import nibabel as nib
import shutil
import tempfile
from pathlib import Path
import os


def is_increment_consistent(num_list):
    """
    Check if the list of numbers is increment consistent.

    Args:
    - num_list (list): List of numbers to check.

    Returns:
    - bool: True if the numbers are increment consistent, False otherwise.
    """

    for i in range(1, len(num_list)):
        if num_list[i] - num_list[i - 1] != 1:
            print(f"num_list[i]: {num_list[i]}, num_list[i - 1]: {num_list[i - 1]}")
            return False  # If the difference between any consecutive pair is not equal to the increment, return False

    return True


def check_slice_increment_consistency(folder_path):
    files = os.listdir(folder_path)
    slice_increment_values = []

    for file_name in files:
        if file_name.endswith('.dcm'):

            num = int(file_name.split('_')[2].split('.')[0])
            slice_increment_values.append(num)

    result = is_increment_consistent(slice_increment_values)
    print(f"{folder_path}: {result}")


def dcm_to_nifti(input_path, output_path):
    """
    Uses dicom2nifti package (also works on windows)

    input_path: a directory of dicom slices
    output_path: a nifti file path
    """
    # dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(str(tmp))

        # convert dicom directory to nifti
        dicom2nifti.convert_directory(input_path, str(tmp),
                                      compression=True, reorient=True)

        #looks for the first NIfTI file (*nii.gz) in temp
        nii = next(tmp.glob('*nii.gz'))

        # copy nifti file to the specified output path and named it 'MRI.nii.gz'
        shutil.copy(nii, output_path +'.nii.gz')


def pci_dicom_to_nifti(input_folder, output_folder):
    count = 0
    for root, dirs, files in os.walk(input_folder):
        # Create the corresponding directory structure in the output folder
        # relative_path = os.path.relpath(root, input_folder)
        # output_path = os.path.join(output_folder, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        for file in files:
            if file.endswith(".dcm"):
                name = f"s{file[1:5]}_000{file[6:7]}"
                converted_data_save_dir = os.path.join(output_folder, name)
                print(f"{converted_data_save_dir}")
                dcm_to_nifti(root, converted_data_save_dir)

                count += 1
                print(f"file {name} have been processed.{count}")
                break  # Add the folder once and move on to the next


def main():

    'convert the image from dicom files to nifti files'
    raw_image_folder = 'E:/graduation_project_TUe/data_from_Lotte/pci_score/PM_scans_all'
    output_folder = 'E:/graduation_project_TUe/data_from_Lotte/pci_score/converted_nii'
    pci_dicom_to_nifti(raw_image_folder, output_folder)


if __name__ == '__main__':

    # folder_path = 'E:/graduation_project_TUe/data_from_Lotte/pci_score/PM_scans_all/'
    # for root, dirs, files in os.walk(folder_path):
    #
    #     for file in files:
    #         if file.endswith(".dcm"):
    #             # folder = os.path.join(folder_path, file)
    #             # check_slice_increment_consistency(root)
    #             validate_slice_increment(root)
    #             break
    main()
