import SimpleITK as sitk
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
from surface_distance.metrics import (compute_surface_distances, compute_robust_hausdorff,
                                      compute_average_surface_distance, compute_surface_dice_at_tolerance,
                                      compute_dice_coefficient)
import os
import time
import pandas as pd

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)


def compute_all_metrics(pred, gt, spacing, bor_gt, bor_pred):
    # Compute the surface distances
    DSC = compute_dice_coefficient(gt, pred)
    border_DSC = compute_dice_coefficient(bor_gt, bor_pred)
    border_clDice = clDice(bor_pred, bor_gt)
    surface_distances = compute_surface_distances(gt, pred, spacing)
    # hausdorff_distance = compute_robust_hausdorff(surface_distances, 100)
    HD95 = compute_robust_hausdorff(surface_distances, 95)
    average_distance = compute_average_surface_distance(surface_distances)
    surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
    surface_dice_2mm = compute_surface_dice_at_tolerance(surface_distances, 2)
    surface_dice_3mm = compute_surface_dice_at_tolerance(surface_distances, 3)

    # round up all the metrics to 4 decimal places
    DSC = round(DSC, 4)
    border_DSC = round(border_DSC, 4)
    border_clDice = round(border_clDice, 4)
    # hausdorff_distance = round(hausdorff_distance, 4)
    HD95 = round(HD95, 4)
    average_distance_gt_to_pred = round(average_distance[0], 4)
    average_distance_pred_to_gt = round(average_distance[1], 4)
    surface_dice_1mm = round(surface_dice_1mm, 4)
    surface_dice_2mm = round(surface_dice_2mm, 4)
    surface_dice_3mm = round(surface_dice_3mm, 4)

    return DSC, border_DSC, border_clDice, HD95, average_distance_pred_to_gt, average_distance_gt_to_pred, surface_dice_1mm, surface_dice_2mm, surface_dice_3mm


def obtain_border_regions(pred, gt, liver_mask, stomach_mask, spleen_mask, pancreas_mask, itk_img):
    """
    :param pred: the prediction image np array
    :param gt: the ground truth image np array
    :param liver_mask: the liver mask np array
    :param stomach_mask: the stomach mask np array
    :param spleen_mask: the spleen mask np array
    :param pancreas_mask: the pancreas mask np array
    :return: the border regions np array
    """
    # create empty np arrays
    border_pred = np.zeros_like(pred)
    border_gt = np.zeros_like(gt)

    # pred mask subtract the organ masks
    r1_border_pred = (pred == 1) - liver_mask
    r2_border_pred = (pred == 2) - liver_mask
    r3_border_pred = (pred == 3) - spleen_mask - stomach_mask - pancreas_mask
    # sign the border regions back
    border_pred[r1_border_pred == 1] = 1
    border_pred[r2_border_pred == 1] = 2
    border_pred[r3_border_pred == 1] = 3

    # gt mask subtract the organ masks
    r1_border_gt = (gt == 1) - liver_mask
    r2_border_gt = (gt == 2) - liver_mask
    r3_border_gt = (gt == 3) - spleen_mask - stomach_mask - pancreas_mask
    # sign the border regions back
    border_gt[r1_border_gt == 1] = 1
    border_gt[r2_border_gt == 1] = 2
    border_gt[r3_border_gt == 1] = 3

    return border_pred, border_gt


def main(prediction_file_path, ground_truth_file_path, dir_organ, case_number):
    # targeted_folder = "./3d_slicer_checker/pred"
    data = []

    sitk_pred = sitk.ReadImage(prediction_file_path)
    sitk_spacing = sitk_pred.GetSpacing()
    array_pred = sitk.GetArrayFromImage(sitk_pred)

    sitk_gt = sitk.ReadImage(ground_truth_file_path)
    array_gt = sitk.GetArrayFromImage(sitk_gt)

    liver_mask = sitk.ReadImage(os.path.join(dir_organs, "liver.nii.gz"))
    stomach_mask = sitk.ReadImage(os.path.join(dir_organs, "stomach.nii.gz"))
    spleen_mask = sitk.ReadImage(os.path.join(dir_organs, "spleen.nii.gz"))
    pancreas_mask = sitk.ReadImage(os.path.join(dir_organs, "pancreas.nii.gz"))
    liver_array = sitk.GetArrayFromImage(liver_mask)
    stomach_array = sitk.GetArrayFromImage(stomach_mask)
    spleen_array = sitk.GetArrayFromImage(spleen_mask)
    pancreas_array = sitk.GetArrayFromImage(pancreas_mask)

    border_pred, border_gt = obtain_border_regions(array_pred, array_gt, liver_array, stomach_array, spleen_array,
                                                   pancreas_array, sitk_pred)

    for i in range(1, 4):
        pred_mask = np.zeros_like(array_pred)
        gt_mask = np.zeros_like(array_gt)
        p_b_mask = np.zeros_like(border_pred)
        g_b_mask = np.zeros_like(border_gt)
        pred_mask[array_pred == i] = 1
        gt_mask[array_gt == i] = 1
        p_b_mask[border_pred == i] = 1
        g_b_mask[border_gt == i] = 1
        array_pred_bool = pred_mask.astype(bool)
        array_gt_bool = gt_mask.astype(bool)
        b_pred_bool = p_b_mask.astype(bool)
        b_gt_bool = g_b_mask.astype(bool)

        (DSC, inter_organ, inter_organ_clDice, HD95, MASD_gt_to_pred, MASD_pred_to_gt, surface_dice_1mm,
         surface_dice_2mm, surface_dice_3mm) = compute_all_metrics(array_pred_bool, array_gt_bool, sitk_spacing, b_gt_bool, b_pred_bool)

        data.append({"Case ID": case_number, "Region_num": i, "DSC": DSC, "Inter-organ DSC": inter_organ,
                     "Inter-organ clDice": inter_organ_clDice, "surface_dice_1mm": surface_dice_1mm,
                     "surface_dice_2mm": surface_dice_2mm, "surface_dice_3mm": surface_dice_3mm,
                     "HD95": HD95,"MASD gt to pred": MASD_gt_to_pred, "MASD pred to gt": MASD_pred_to_gt})

    avg_DSC = (data[0]["DSC"] + data[1]["DSC"] + data[2]["DSC"]) / 3
    avg_border_DSC = (data[0]["Inter-organ DSC"] + data[1]["Inter-organ DSC"] + data[2]["Inter-organ DSC"]) / 3
    avg_border_clDice = (data[0]["Inter-organ clDice"] + data[1]["Inter-organ clDice"] + data[2]["Inter-organ clDice"]) / 3
    # avg_hausdorff_distance = (data[0]["hausdorff_distance"] + data[1]["hausdorff_distance"] + data[2]["hausdorff_distance"]) / 3
    avg_HD95 = (data[0]["HD95"] + data[1]["HD95"] + data[2]["HD95"]) / 3
    avg_MASD_gt_to_pred = (data[0]["MASD gt to pred"] + data[1]["MASD gt to pred"] + data[2]["MASD gt to pred"]) / 3
    avg_MASD_pred_to_gt = (data[0]["MASD pred to gt"] + data[1]["MASD pred to gt"] + data[2]["MASD pred to gt"]) / 3
    avg_surface_dice_1mm = (data[0]["surface_dice_1mm"] + data[1]["surface_dice_1mm"] + data[2]["surface_dice_1mm"]) / 3
    avg_surface_dice_2mm = (data[0]["surface_dice_2mm"] + data[1]["surface_dice_2mm"] + data[2]["surface_dice_2mm"]) / 3
    avg_surface_dice_3mm = (data[0]["surface_dice_3mm"] + data[1]["surface_dice_3mm"] + data[2]["surface_dice_3mm"]) / 3

    data.append({"Case ID": case_number, "Region_num": "average of 3 regions", "DSC": round(avg_DSC, 4),
                 "Inter-organ DSC": round(avg_border_DSC, 4), "Inter-organ clDice": round(avg_border_clDice, 4),
                 "surface_dice_1mm": round(avg_surface_dice_1mm, 4), "surface_dice_2mm": round(avg_surface_dice_2mm, 4),
                 "surface_dice_3mm": round(avg_surface_dice_3mm, 4), "HD95": round(avg_HD95, 4),
                 "MASD gt to pred": round(avg_MASD_gt_to_pred, 4), "MASD pred to gt": round(avg_MASD_pred_to_gt, 4)})
    return data


if __name__ == "__main__":
    start_time = time.time()
    '''
    following code for test single case
    '''
    # pred_path = "./code_test_folder/3d_slicer_checker/pred/s0158.nii.gz"
    # gt_path = "./code_test_folder/3d_slicer_checker/gt/s0158.nii.gz"
    # dir_organs = "./code_test_folder/3d_slicer_checker/organs_masks/s0158"
    # data = main(pred_path, gt_path, dir_organs, case_number="s0158")


    '''
    following code for test multiple cases in the input folder
    '''
    input_folder = "./code_test_folder/for_evaluation/"
    pred_folder = "SimLowRes"

    # pred_folder = "gaussianBlur"
    # pred_folder = "gaussianNoise"
    # pred_folder = "no_DA_200epoch"
    # pred_folder = "R_S_Sim"
    # pred_folder = "rotation"
    # pred_folder = "scale"
    # pred_folder = "SimLowRes"

    pred_path = os.path.join(input_folder, f"200epochs/{pred_folder}")
    gt_path = os.path.join(input_folder, "gt")
    organs_masks_path = os.path.join(input_folder, "organs_masks")
    count = 0
    df = pd.DataFrame()
    for file in os.listdir(pred_path):
        if file.endswith(".nii.gz"):
            count += 1
            forloop_start_time = time.time()
            prediction_file_path = os.path.join(pred_path, file)
            ground_truth_file_path = os.path.join(gt_path, file)
            dir_organs = os.path.join(organs_masks_path, file.split(".")[0])
            data = main(prediction_file_path, ground_truth_file_path, dir_organs, case_number=file.split(".")[0])
            if count == 1:
                # Append data to the DataFrame
                df = df._append(pd.DataFrame(data), ignore_index=False)
            else:
                df = df._append(pd.DataFrame(data), ignore_index=True)
            print(f"file {file} is done!")
            print("--- %s seconds ---" % round(time.time() - forloop_start_time, 2))

            # if count == 2:
            #     break

    # Plot data in bar chart for better visualization
    # plot_bar(df)

    # Save the DataFrame to a csv file
    df.to_csv(f"{input_folder}evaluation_metrics_{pred_folder}.csv", index=False)
    print("--- Total time %s seconds ---" % round(time.time() - start_time, 2))
