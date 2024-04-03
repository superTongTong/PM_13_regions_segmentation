import SimpleITK as sitk
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def crop_image(img_array, mask_array):
    '''
    :param img_array np.array
    :param mask_array np.array
    :return: cropped_img_array
    '''
    # Find the indices of nonzero values
    nonzero_indices = np.argwhere(mask_array)

    if len(nonzero_indices) == 0:
        # No nonzero values found
        print("No nonzero values found in the input array")
        return None

    # Extract the minimum and maximum indices along each axis
    min_indices = np.min(nonzero_indices, axis=0)
    max_indices = np.max(nonzero_indices, axis=0)

    # Create the sub-array based on the boundary defined by nonzero values
    sub_array = img_array[min_indices[0]:max_indices[0],
                            min_indices[1]:max_indices[1],
                            min_indices[2]:max_indices[2]]

    return sub_array


def extract_roi_based_mask(ct_image_path, seg_mask_path, idx):
    # Load the CT image and segmentation mask
    ct_image = sitk.ReadImage(ct_image_path, sitk.sitkInt32)
    seg_mask = sitk.ReadImage(seg_mask_path, sitk.sitkInt32)

    # Find bounding box of segmentation mask
    region = sitk.LabelShapeStatisticsImageFilter()
    region.Execute(seg_mask)
    # Get bounding box of region of interest, [x_start, y_start, z_start,x_size, y_size, z_size].
    bounding_box = region.GetBoundingBox(idx)

    # Crop the CT image using the bounding box
    ct_image_cropped = sitk.RegionOfInterest(ct_image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    ct_mask_cropped = sitk.RegionOfInterest(seg_mask, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

    # Convert segmentation mask to NumPy array
    seg_mask_arr = sitk.GetArrayFromImage(ct_mask_cropped)

    # Apply binary mask to CT image
    ct_image_arr = sitk.GetArrayFromImage(ct_image_cropped)
    roi_ct_image_arr = ct_image_arr * seg_mask_arr

    # Create SimpleITK image from the extracted ROI array
    roi_ct_image = sitk.GetImageFromArray(roi_ct_image_arr)
    roi_ct_image.CopyInformation(ct_image_cropped)

    return roi_ct_image #ct_image_cropped # roi_ct_image

def crop_based_mask(ct_image_path, seg_mask_path, idx):
    # Load the CT image and segmentation mask
    ct_image = sitk.ReadImage(ct_image_path, sitk.sitkFloat64)
    seg_mask = sitk.ReadImage(seg_mask_path, sitk.sitkInt8)

    # Find bounding box of segmentation mask
    region = sitk.LabelShapeStatisticsImageFilter()
    region.Execute(seg_mask)
    # Get bounding box of region of interest, [x_start, y_start, z_start,x_size, y_size, z_size].
    bounding_box = region.GetBoundingBox(idx)

    # Crop the CT image using the bounding box
    ct_image_cropped = sitk.RegionOfInterest(ct_image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

    return ct_image_cropped


def mian():
    # data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/images'
    # mask_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/masks'
    # save_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan'
    # csv_path = '/data/data_ViT/PCI_3_regions.csv'
    data_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/raw_data/'
    mask_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/masks/'
    save_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/cropped_scan'
    csv_path = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/PCI_3_regions_new.csv'
    random_seed = 42
    df = pd.read_csv(csv_path)


    img_list = os.listdir(data_dir)

    train_list, valid_list = train_test_split(img_list,
                                              test_size=0.2,
                                              random_state=random_seed)

    for img_name in train_list:
        save_train_dir = os.path.join(save_dir, 'train')
        os.makedirs(save_train_dir, exist_ok=True)
        case_PCI = df[df['CaseID'] == img_name.split('.')[0]]

        mask_path = os.path.join(mask_dir, img_name)
        img_path = os.path.join(data_dir, img_name)

        for i in range(1, 4):
            region_score = case_PCI[f'R{i}'].values
            sc = region_score[0]
            crop_img = extract_roi_based_mask(img_path, mask_path, i)
            name = img_name.split('.')[0]
            sitk.WriteImage(crop_img, f"{save_train_dir}/{name}_R{i}_{sc}.nii.gz")
        # break
    for img_name in valid_list:
        save_val_dir = os.path.join(save_dir, 'validation')
        os.makedirs(save_val_dir, exist_ok=True)
        case_PCI = df[df['CaseID'] == img_name.split('.')[0]]

        mask_path = os.path.join(mask_dir, img_name)
        img_path = os.path.join(data_dir, img_name)

        for i in range(1, 4):
            region_score = case_PCI[f'R{i}'].values
            sc = region_score[0]
            crop_img = extract_roi_based_mask(img_path, mask_path, i)
            name = img_name.split('.')[0]
            sitk.WriteImage(crop_img, f"{save_val_dir}/{name}_R{i}_{sc}.nii.gz")

if __name__ == "__main__":
    mian()
