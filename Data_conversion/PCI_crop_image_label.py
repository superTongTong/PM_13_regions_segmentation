import SimpleITK as sitk
import os


def extract_roi(ct_image_path, seg_mask_path, idx):
    # Load the CT image and segmentation mask
    ct_image = sitk.ReadImage(ct_image_path, sitk.sitkInt32)
    seg_mask = sitk.ReadImage(seg_mask_path, sitk.sitkInt8)

    # Find bounding box of segmentation mask
    region = sitk.LabelShapeStatisticsImageFilter()
    region.Execute(seg_mask)
    # Get bounding box of region of interest, [x_start, y_start, z_start,x_size, y_size, z_size].
    bounding_box = region.GetBoundingBox(idx)

    # Crop the CT image using the bounding box
    ct_image_cropped = sitk.RegionOfInterest(ct_image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

    return ct_image_cropped # roi_ct_image


def mian():
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/images'
    mask_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/masks'
    save_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/'
    os.makedirs(save_dir, exist_ok=True)
    img_list = os.listdir(data_dir)

    for img_name in img_list:

        mask_path = os.path.join(mask_dir, img_name)
        img_path = os.path.join(data_dir, img_name)

        for i in range(1, 4):
            crop_img = extract_roi(img_path, mask_path, i)
            name = img_name.split('.')[0]
            sitk.WriteImage(crop_img, f"{save_dir}/{name}_R{i}.nii.gz")


if __name__ == "__main__":
    mian()
