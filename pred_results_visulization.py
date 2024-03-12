import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from Figure_3D_image import plot_3d_multi


def create_seg_figure(background, color_seg, color_seg_pred, color_seg_diff, slice_idx, sample_id):
    background = np.transpose(background)
    color_seg = np.transpose(color_seg, (1, 0, 2))#Anterior view
    color_seg_pred = np.transpose(color_seg_pred, (1, 0, 2)) #Anterior view
    color_seg_diff = np.transpose(color_seg_diff, (1, 0, 2))#Anterior view
    alpha = 0.3
    plt.figure(figsize=(10, 10))

    # prediction
    plt.subplot(1, 3, 1)
    plt.imshow(background, cmap='soft_tissue')
    plt.imshow(color_seg_pred, cmap='bone', alpha=alpha)
    plt.title("prediction", fontsize=20)
    plt.axis('off')
    plt.tight_layout()

    # case and slice id
    subtext = f"Case_{sample_id}"
    slice_txt = f"{slice_idx:03d}"
    plt.text(120, 18, subtext, fontsize=20, color='white')
    plt.text(5, 20, slice_txt, fontsize=30, color='white')

    # target
    plt.subplot(1, 3, 2)
    plt.imshow(background, cmap='soft_tissue')
    plt.imshow(color_seg, cmap='bone', alpha=alpha)
    plt.title("target", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.text(120, 18, subtext, fontsize=20, color='white')
    plt.text(5, 20, slice_txt, fontsize=30, color='white')

    # non-overlapped region
    plt.subplot(1, 3, 3)
    plt.imshow(background, cmap='soft_tissue')
    plt.imshow(color_seg_diff, cmap='bone', alpha=alpha)
    plt.title("non-overlapped_region", fontsize=20)
    plt.axis('off')
    plt.tight_layout()

    # case and slice id
    subtext = f"Case_{sample_id}"
    slice_txt = f"{slice_idx:03d}"
    plt.text(120, 18, subtext, fontsize=20, color='white')
    plt.text(5, 20, slice_txt, fontsize=30, color='white')
    # plt.show()
    return plt


def check_plot(background, diff_seg):
    ############################
    # source: https://theaisummer.com/medical-image-python/
    # level = 50, window = 250 for soft tissue
    level = 50
    window = 250
    background = np.transpose(background)
    max = level + window / 2
    min = level - window / 2
    background = background.clip(min, max)
    ############################
    diff_seg = np.transpose(diff_seg, (1, 0, 2))
    alpha = 0.3
    plt.figure(figsize=(10, 10))
    plt.imshow(background, cmap='gray')
    plt.imshow(diff_seg, cmap='bone', alpha=alpha)
    plt.title("prediction", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return plt


def create_gif_prediction(sample_id, background, color_segmentation, color_segmentation_pred, color_diff, num_layer):

    case_path = f'./saved_gifs/case_{sample_id}_3'
    if not os.path.exists(case_path):
        os.makedirs(case_path)

    images = []  # to store images for the GIF
    # store all the images as png
    for layer in range(100, 120):
    # for layer in range(30, num_layer - 50):
        image = create_seg_figure(background[:, layer, :], color_segmentation[:, layer, :, :],
                                  color_segmentation_pred[:, layer, :, :], color_diff[:, layer, :, :], layer, sample_id)

        # save image
        image_path = f'{case_path}/case_{sample_id}_{layer:03d}.png'
        plt.savefig(image_path)

        # append image path to the list
        images.append(imageio.imread(image_path))

        # close the figure to release resources
        image.close()

    # create GIF
    gif_path = f'{case_path}/case_{sample_id}_animation.gif'
    imageio.mimsave(gif_path, images)


def check_orientation(ct_image, ct_arr):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'L':
        ct_arr = np.flip(ct_arr, axis=0)
    if y != 'P':
        ct_arr = np.flip(ct_arr, axis=1)
    if z != 'I':
        ct_arr = np.flip(ct_arr, axis=2)

    return ct_arr


def load_data(data_path):

    # load data
    img = nib.load(data_path)
    RAS_img = check_orientation(img, img.get_fdata())

    return RAS_img


def find_non_overlap(segmentation_pred, segmentation_gt):
    """
    Find the non-overlap between the target and the prediction
    :param color_segmentation_pred: array
    :param color_segmentation: array
    :return: array
    """
    overlap_seg = np.zeros_like(segmentation_pred)
    gt = segmentation_gt
    pred = segmentation_pred
    for i in range(1, 4):
        overlapped_mask = ((gt == i) & (pred == i).astype(int))
        union_mask = ((gt == i) | (pred == i)).astype(int)
        differ_mask = union_mask - overlapped_mask
        overlap_seg[differ_mask == 1] = i

    return overlap_seg


def run_process(image_path, seg_path, target_path, sample_id):
    '''
     0: 'background',
     1: 'Red = region 1',
     2: 'Green = region 2',
     3: 'Blue = region 3'
     '''

    #load data
    background = load_data(image_path)
    seg_pred = load_data(seg_path)
    seg_target = load_data(target_path)
    a = seg_pred.shape[0]
    b = seg_pred.shape[1]
    num_layer = seg_pred.shape[2]

    # change colours of segmentation result
    color_segmentation = np.zeros((a, b, num_layer, 3), dtype=np.uint8)
    color_segmentation_pred = np.zeros((a, b, num_layer, 3), dtype=np.uint8)
    color_segmentation_dif = np.zeros((a, b, num_layer, 3), dtype=np.uint8)

    # find non-overlaped region between target and prediction
    # differ_seg = find_non_overlap(seg_pred, seg_target)
    # background = 0
    color_segmentation[seg_target == 1] = [255, 0, 0]  # Red
    color_segmentation[seg_target == 2] = [0, 255, 0]  # Green
    color_segmentation[seg_target == 3] = [0, 204, 255]  # Blue

    # # background = 0
    color_segmentation_pred[seg_pred == 1] = [255, 0, 0]  # Red
    color_segmentation_pred[seg_pred == 2] = [0, 255, 0]  # Green
    color_segmentation_pred[seg_pred == 3] = [0, 204, 255]  # light blue

    # background = 0
    # color_segmentation_dif[differ_seg == 1] = [255, 0, 0]  # Red
    # color_segmentation_dif[differ_seg == 2] = [0, 255, 0]  # Green
    # color_segmentation_dif[differ_seg == 3] = [0, 0, 255]  # Blue
    #################################################
    # # Create masks for exclusive areas
    # exclusive_gt = np.logical_and(seg_target != seg_pred, seg_target != 0) # gt mask > pred mask
    # exclusive_pred = np.logical_and(seg_pred != seg_target, seg_pred != 0) # pred mask > gt mask
    # # Set colors for exclusive areas
    # color_segmentation_dif[exclusive_gt] = [255, 0, 255]  # pink for gt mask > pred mask
    # color_segmentation_dif[exclusive_pred] = [0, 255, 255]  # light aqua for pred mask > gt mask
    #################################################
    # Create masks for exclusive areas for each region
    exclusive_gt_1 = np.logical_and(seg_target == 1, seg_target != seg_pred)
    exclusive_gt_2 = np.logical_and(seg_target == 2, seg_target != seg_pred)
    exclusive_gt_3 = np.logical_and(seg_target == 3, seg_target != seg_pred)

    exclusive_pred_1 = np.logical_and(seg_pred == 1, seg_pred != seg_target)
    exclusive_pred_2 = np.logical_and(seg_pred == 2, seg_pred != seg_target)
    exclusive_pred_3 = np.logical_and(seg_pred == 3, seg_pred != seg_target)

    # Set colors for exclusive areas
    # color_segmentation_dif[exclusive_gt_1] = [255, 69, 0]  # Orange for exclusive to ground truth, region 1
    # color_segmentation_dif[exclusive_gt_2] = [255, 69, 0]  # Orange for exclusive to ground truth, region 2
    # color_segmentation_dif[exclusive_gt_3] = [255, 69, 0]  # Orange for exclusive to ground truth, region 3
    #
    # color_segmentation_dif[exclusive_pred_1] = [173, 216, 230]  # Light Blue for exclusive to prediction, region 1
    # color_segmentation_dif[exclusive_pred_2] = [173, 216, 230]  # Light Blue for exclusive to prediction, region 2
    # color_segmentation_dif[exclusive_pred_3] = [173, 216, 230]  # Light Blue for exclusive to prediction, region 3
    #################################################
    # save segmentation result as gif

    # check_plot(background[:, 118, :],  color_segmentation_pred[:, 118, :])
    plot_3d_multi(exclusive_gt_1, exclusive_gt_2, exclusive_gt_3, threshold=0.5, elev=180, azim=0)
    # create_gif_prediction(sample_id, background, color_segmentation, color_segmentation_pred, color_segmentation_dif, b)


if __name__ == '__main__':
    # folder_in = './3d_slicer_checker/images'
    # for file in os.listdir(folder_in):
    #     if file.endswith(".nii.gz"):
    #         sample_id = file.split('_')[0]
    #         image_path = f'./3d_slicer_checker/images/{sample_id}_0000.nii.gz'
    #         seg_path = f'./3d_slicer_checker/pred/pred_{sample_id}.nii.gz'
    #         target_path = f'./3d_slicer_checker/gt/{sample_id}.nii.gz'
    #         run_process(image_path, seg_path, target_path, sample_id)

    '''
    this is for testing the function
    '''
    sample_id = 's0046'
    image_path = f'./code_test_folder/3d_slicer_checker/images/{sample_id}_0000.nii.gz'
    seg_path = f'./code_test_folder/3d_slicer_checker/pred/{sample_id}.nii.gz'
    target_path = f'./code_test_folder/3d_slicer_checker/gt/{sample_id}.nii.gz'
    run_process(image_path, seg_path, target_path, sample_id)