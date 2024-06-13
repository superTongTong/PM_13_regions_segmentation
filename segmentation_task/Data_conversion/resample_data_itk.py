import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


def resample_img(itk_image, out_spacing=1.5, is_label=False):
    # Resample images to 1.5mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    final_spacing = [out_spacing, out_spacing, out_spacing]
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / final_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / final_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / final_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(final_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def process_files_in_folder(data_path, output_path):
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
        resampled_sitk_img = resample_img(itk_image, is_label=False)

        # Write the resampled image
        sitk.WriteImage(resampled_sitk_img, os.path.join(output_path, f))


def test_one_file(data_path):

    itk_label = sitk.ReadImage(data_path)
    # Assume to have some sitk image (itk_image) and label (itk_label)
    # resampled_sitk_img = resample_img(itk_image, is_label=False)
    resampled_sitk_lbl = resample_img(itk_label, is_label=True)
    sitk.WriteImage(resampled_sitk_lbl, '../code_test_folder/test_image_label/s0046_resampled.nii.gz')


if __name__ == "__main__":

    in_path = '../data/nnunet/raw/Dataset007_CKI_DATA/imagesTr'
    out_path = '../data/nnunet/raw/Dataset007_CKI_DATA/imagesTr'
    os.makedirs(out_path, exist_ok=True)

    process_files_in_folder(in_path, out_path)
