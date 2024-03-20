import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
import pandas as pd


def create_seg_figure(background, color_seg, color_seg_pred, r1_diff, r2_diff, r3_diff, slice_idx, sample_id):
    ############################
    # source: https://theaisummer.com/medical-image-python/
    # level = 50, window = 250 for soft tissue
    level = 50
    window = 250
    background = np.transpose(background)
    max = level + window / 2
    min = level - window / 2
    background = background.clip(min, max)

    df_legend = pd.DataFrame([[0, [255, 0, 0], 'R1'],
                              [1, [0, 255, 0], 'R2'],
                              [2, [0, 0, 255], 'R3'],
                              [3, [0, 255, 255], 'R1 exclusive to prediction'],
                              [4, [255, 255, 0], 'R1 exclusive to ground truth'],
                              [5, [255, 127, 80], 'R2 exclusive to prediction'],
                              [6, [255, 0, 255], 'R2 exclusive to ground truth'],
                              [7, [50, 205, 50], 'R3 exclusive to prediction'],
                              [8, [218, 112, 214], 'R3 exclusive to ground truth']],
                             columns=['key', 'color', 'name'])

    handles_1 = [Rectangle((50, 100), 5, 5, color=[c / 255 for c in color_list]) for color_list in df_legend['color']]

    labels = df_legend['name']
    ############################
    color_seg = np.transpose(color_seg, (1, 0, 2)) # Anterior view
    color_seg_pred = np.transpose(color_seg_pred, (1, 0, 2)) # Anterior view
    # color_seg_diff = np.transpose(color_seg_diff, (1, 0, 2))# Anterior view
    r1_diff = np.transpose(r1_diff, (1, 0, 2)) # Anterior view
    r2_diff = np.transpose(r2_diff, (1, 0, 2))  # Anterior view
    r3_diff = np.transpose(r3_diff, (1, 0, 2))  # Anterior view
    alpha = 0.5
    plt.figure(figsize=(25, 20))

    # prediction
    plt.subplot(2, 3, 1)
    plt.imshow(background, cmap='gray')
    plt.imshow(color_seg_pred, cmap='bone', alpha=alpha)
    plt.title("Model Prediction", fontsize=30)
    plt.axis('off')

    # case and slice id
    subtext = f"Case_{sample_id}"
    slice_txt = f"{slice_idx:03d}"
    plt.text(2, 10, subtext, fontsize=15, color='white')
    plt.text(2, 30, slice_txt, fontsize=15, color='white')

    # target
    plt.subplot(2, 3, 2)
    plt.imshow(background, cmap='gray')
    plt.imshow(color_seg, cmap='bone', alpha=alpha)
    plt.title("Ground Truth", fontsize=30)
    plt.axis('off')

    # plt.text(190, 18, subtext, fontsize=10, color='white')
    # plt.text(5, 20, slice_txt, fontsize=15, color='white')

    # error over 3 region
    plt.subplot(2, 3, 3)
    plt.rcParams.update({'legend.fontsize': 26})
    plt.legend(handles_1, labels, mode='expand', ncol=1, loc='center', facecolor='gray')
    plt.axis('off')

    # plt.imshow(background, cmap='gray')
    # plt.imshow(color_seg_diff, cmap='bone', alpha=alpha)
    # plt.title("Error over 3 regions", fontsize=30)
    # plt.axis('off')

    # # case and slice id
    # subtext = f"Case_{sample_id}"
    # slice_txt = f"{slice_idx:03d}"
    # plt.text(190, 18, subtext, fontsize=10, color='white')
    # plt.text(5, 20, slice_txt, fontsize=15, color='white')

    # error over R1
    plt.subplot(2, 3, 4)
    plt.imshow(background, cmap='gray')
    plt.imshow(r1_diff, cmap='bone', alpha=alpha)
    plt.title("Error region 1", fontsize=30)
    plt.axis('off')

    # # case and slice id
    # subtext = f"Case_{sample_id}"
    # slice_txt = f"{slice_idx:03d}"
    # plt.text(190, 18, subtext, fontsize=10, color='white')
    # plt.text(5, 20, slice_txt, fontsize=15, color='white')

    # error over R2
    plt.subplot(2, 3, 5)
    plt.imshow(background, cmap='gray')
    plt.imshow(r2_diff, cmap='bone', alpha=alpha)
    plt.title("Error region 2", fontsize=30)
    plt.axis('off')

    # # case and slice id
    # subtext = f"Case_{sample_id}"
    # slice_txt = f"{slice_idx:03d}"
    # plt.text(190, 18, subtext, fontsize=10, color='white')
    # plt.text(5, 20, slice_txt, fontsize=15, color='white')

    # error over R3
    plt.subplot(2, 3, 6)
    plt.imshow(background, cmap='gray')
    plt.imshow(r3_diff, cmap='bone', alpha=alpha)
    plt.title("Error region 3", fontsize=30)
    plt.axis('off')

    # # case and slice id
    # subtext = f"Case_{sample_id}"
    # slice_txt = f"{slice_idx:03d}"
    # plt.text(190, 18, subtext, fontsize=10, color='white')
    # plt.text(5, 20, slice_txt, fontsize=15, color='white')

    plt.tight_layout()
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
    alpha = 0.5
    plt.figure(figsize=(10, 10))
    plt.imshow(background, cmap='gray')
    plt.imshow(diff_seg, cmap='bone', alpha=alpha)
    plt.title("prediction", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return plt


def create_gif_prediction(sample_id, background, color_segmentation, color_segmentation_pred, r1_diff, r2_diff, r3_diff, num_layer):

    case_path = f'./code_test_folder/saved_gifs/region_based_diff_v2/case_{sample_id}'
    if not os.path.exists(case_path):
        os.makedirs(case_path)

    # images = []  # to store images for the GIF
    # store all the images as png
    for layer in range(30, num_layer-35):
    # for layer in range(30, num_layer - 50):
        image = create_seg_figure(background[:, layer, :], color_segmentation[:, layer, :, :],
                                  color_segmentation_pred[:, layer, :, :], r1_diff[:, layer, :, :],
                                  r2_diff[:, layer, :, :], r3_diff[:, layer, :, :], layer, sample_id)

        # save image
        image_path = f'{case_path}/case_{sample_id}_{layer:03d}.png'
        plt.savefig(image_path)

        # append image path to the list
        # images.append(imageio.imread(image_path))

        # close the figure to release resources
        image.close()

    # create GIF
    # gif_path = f'{case_path}/case_{sample_id}_animation.gif'
    # imageio.mimsave(gif_path, images)


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
    R1_dif = np.zeros((a, b, num_layer, 3), dtype=np.uint8)
    R2_dif = np.zeros((a, b, num_layer, 3), dtype=np.uint8)
    R3_dif = np.zeros((a, b, num_layer, 3), dtype=np.uint8)

    # find non-overlaped region between target and prediction
    # differ_seg = find_non_overlap(seg_pred, seg_target)
    # background = 0
    color_segmentation[seg_target == 1] = [255, 0, 0]  # Red
    color_segmentation[seg_target == 2] = [0, 255, 0]  # Green
    color_segmentation[seg_target == 3] = [0, 0, 255]  # Blue

    # # background = 0
    color_segmentation_pred[seg_pred == 1] = [255, 0, 0]  # Red
    color_segmentation_pred[seg_pred == 2] = [0, 255, 0]  # Green
    color_segmentation_pred[seg_pred == 3] = [0, 0, 255]  # Blue

    #################################################
    # Create masks for exclusive areas for each region
    exclusive_gt_1 = np.logical_and(seg_target == 1, seg_target != seg_pred)
    exclusive_gt_2 = np.logical_and(seg_target == 2, seg_target != seg_pred)
    exclusive_gt_3 = np.logical_and(seg_target == 3, seg_target != seg_pred)

    exclusive_pred_1 = np.logical_and(seg_pred == 1, seg_pred != seg_target)
    exclusive_pred_2 = np.logical_and(seg_pred == 2, seg_pred != seg_target)
    exclusive_pred_3 = np.logical_and(seg_pred == 3, seg_pred != seg_target)

    # # Set colors for exclusive areas
    # color_segmentation_dif[exclusive_pred_1] = [0, 255, 255]  # Cyan for exclusive to ground truth, region 1
    # color_segmentation_dif[exclusive_pred_2] = [255, 127, 80]  # Coral for exclusive to ground truth, region 2
    # color_segmentation_dif[exclusive_pred_3] = [0, 191, 255]  # Deep blue for exclusive to ground truth, region 3
    #
    # color_segmentation_dif[exclusive_gt_1] = [255, 255, 0]  # Yellow for exclusive to prediction, region 1
    # color_segmentation_dif[exclusive_gt_2] = [255, 0, 255]  # Magenta for exclusive to prediction, region 2
    # color_segmentation_dif[exclusive_gt_3] = [218, 112, 214]  # Orchid for exclusive to prediction, region 3
    #################################################
    R1_dif[exclusive_pred_1] = [0, 255, 255]  # Cyan for exclusive to ground truth, region 1
    R1_dif[exclusive_gt_1] = [255, 255, 0]  # Yellow for exclusive to prediction, region 1

    R2_dif[exclusive_pred_2] = [255, 127, 80]  # Coral for exclusive to ground truth, region 2
    R2_dif[exclusive_gt_2] = [255, 0, 255]  # Magenta for exclusive to prediction, region 2

    R3_dif[exclusive_pred_3] = [50, 205, 50]  # Lime green for exclusive to ground truth, region 3
    R3_dif[exclusive_gt_3] = [218, 112, 214]  # Orchid for exclusive to prediction, region 3


    # save segmentation result as gif
    # check_plot(background[:, 70, :],  color_segmentation_dif[:, 70, :])
    # create_seg_figure(background[:, 120, :], color_segmentation[:, 120, :], color_segmentation_pred[:, 120, :],
    #                   color_segmentation_dif[:, 120, :], R1_dif[:, 120, :], R2_dif[:, 120, :], R3_dif[:, 120, :], slice_idx=1, sample_id=1)
    # plot_3d_multi(exclusive_pred_1, exclusive_pred_2, exclusive_pred_3, exclusive_gt_1, exclusive_gt_2, exclusive_gt_3, threshold=0, elev=180, azim=0)
    create_gif_prediction(sample_id, background, color_segmentation, color_segmentation_pred, R1_dif, R2_dif, R3_dif, b)


if __name__ == '__main__':
    folder_in = './code_test_folder/3d_slicer_checker/images'
    for file in os.listdir(folder_in):

        if file.endswith(".nii.gz"):
            sample_id = file.split('_')[0]
            image_path = f'./code_test_folder/3d_slicer_checker/images/{sample_id}_0000.nii.gz'
            seg_path = f'./code_test_folder/3d_slicer_checker/pred/{sample_id}.nii.gz'
            target_path = f'./code_test_folder/3d_slicer_checker/gt/{sample_id}.nii.gz'
            run_process(image_path, seg_path, target_path, sample_id)

    # '''
    # this is for testing the function
    # '''
    # sample_id = 's0059'
    # image_path = f'./code_test_folder/3d_slicer_checker/images/{sample_id}_0000.nii.gz'
    # seg_path = f'./code_test_folder/3d_slicer_checker/pred/{sample_id}.nii.gz'
    # target_path = f'./code_test_folder/3d_slicer_checker/gt/{sample_id}.nii.gz'
    # run_process(image_path, seg_path, target_path, sample_id)
