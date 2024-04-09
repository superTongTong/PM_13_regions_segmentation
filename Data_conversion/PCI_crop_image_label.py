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
    ct_image1 = sitk.ReadImage(ct_image_path, sitk.sitkInt32)
    seg_mask = sitk.ReadImage(seg_mask_path, sitk.sitkInt32)

    # scale the HU values of ct_img, HU windowing for abdomen CT images: [-200, 300] to [0, 1]
    clip_img = sitk.Clamp(ct_image1, lowerBound=-200, upperBound=300)

    # Find bounding box of segmentation mask
    region = sitk.LabelShapeStatisticsImageFilter()
    region.Execute(seg_mask)
    # Get bounding box of region of interest, [x_start, y_start, z_start,x_size, y_size, z_size].
    bounding_box = region.GetBoundingBox(idx)

    # Crop the CT image using the bounding box
    ct_image_cropped = sitk.RegionOfInterest(clip_img, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    ct_mask_cropped = sitk.RegionOfInterest(seg_mask, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

    # Convert segmentation mask to NumPy array

    seg_mask_arr = sitk.GetArrayFromImage(ct_mask_cropped)
    seg = np.zeros_like(seg_mask_arr)
    seg[seg_mask_arr==idx] = 1

    # Apply binary mask to CT image
    ct_image_arr = sitk.GetArrayFromImage(ct_image_cropped)
    ct_image_arr = (ct_image_arr+200)/500
    roi_ct_image_arr = ct_image_arr * seg

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
    # save_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test'
    # csv_path = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/PCI_3_regions_new.csv'
    data_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/raw_data/'
    mask_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/masks/'
    save_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/cropped_scan_v2'
    csv_path = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/PCI_3_regions_new.csv'
    # random_seed = 42
    df = pd.read_csv(csv_path)
    train_dir = os.path.join(data_dir, 'train')
    val_data_dir = os.path.join(data_dir, 'val')
    # train_list, valid_list = train_test_split(img_list,
    #                                           test_size=0.2,
    #                                           random_state=random_seed)
    train_list = os.listdir(train_dir)
    valid_list = os.listdir(val_data_dir)

    for img_name in train_list:
        print('-'*10)
        print('start processing image:', img_name.split('.')[0])
        save_train_dir = os.path.join(save_dir, 'train')
        os.makedirs(save_train_dir, exist_ok=True)
        # check if the image is already processed
        processed_files_train = os.listdir(save_train_dir)
        # if any files in the save_train_dir has the same caseID as the current image, skip the image
        if len(processed_files_train) > 0 and any([img_name.split('_')[0] in file for file in processed_files_train]):
            print('already processed image:', img_name.split('.')[0])
            continue
        else:
            case_PCI = df[df['CaseID'] == img_name.split('.')[0]]
            # print('PCI_info:', case_PCI)
            mask_d = img_name.replace('_0001', '')
            mask_path = os.path.join(mask_dir, mask_d)
            img_path = os.path.join(train_dir, img_name)
            for i in range(1, 4):
                region_score = case_PCI[f'R{i}'].values
                # print('region score:', region_score)
                sc = region_score[0]
                crop_img = extract_roi_based_mask(img_path, mask_path, i)
                name = img_name.split('.')[0]
                sitk.WriteImage(crop_img, f"{save_train_dir}/{name}_R{i}_{sc}.nii.gz")
            print('finish processing image:', img_name.split('.')[0])
        # break

    for img_name in valid_list:
        print('-' * 10)
        print('start processing image:', img_name.split('.')[0])
        save_val_dir = os.path.join(save_dir, 'validation')
        os.makedirs(save_val_dir, exist_ok=True)
        processed_files_val = os.listdir(save_val_dir)
        # if any files in the save_train_dir has the same caseID as the current image, skip the image
        if len(processed_files_val) > 0 and any([img_name.split('_')[0] in file for file in processed_files_val]):
            print('already processed image:', img_name.split('.')[0])
            continue
        else:
            case_PCI = df[df['CaseID'] == img_name.split('.')[0]]

            mask_d = img_name.replace('_0001', '')
            mask_path = os.path.join(mask_dir, mask_d)
            img_path = os.path.join(val_data_dir, img_name)

            for i in range(1, 4):
                region_score = case_PCI[f'R{i}'].values
                sc = region_score[0]
                crop_img = extract_roi_based_mask(img_path, mask_path, i)
                name = img_name.split('.')[0]
                sitk.WriteImage(crop_img, f"{save_val_dir}/{name}_R{i}_{sc}.nii.gz")
            print('finish processing image:', img_name.split('.')[0])
        # break

if __name__ == "__main__":
    mian()
