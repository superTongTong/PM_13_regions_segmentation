import SimpleITK as sitk
import numpy as np
from surface_distance.metrics import (compute_surface_distances, compute_robust_hausdorff,
                                      compute_average_surface_distance, compute_surface_dice_at_tolerance,
                                      compute_dice_coefficient)

import csv
import os
def compute_all_metrics(pred, gt, spacing):
    # Compute the surface distances
    DSC = compute_dice_coefficient(gt, pred)
    surface_distances = compute_surface_distances(gt, pred, spacing)
    hausdorff_distance = compute_robust_hausdorff(surface_distances, 100)
    HD95 = compute_robust_hausdorff(surface_distances, 95)
    average_distance = compute_average_surface_distance(surface_distances)
    surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
    surface_dice_2mm = compute_surface_dice_at_tolerance(surface_distances, 2)
    surface_dice_3mm = compute_surface_dice_at_tolerance(surface_distances, 3)

    return DSC, hausdorff_distance, HD95, average_distance, surface_dice_1mm, surface_dice_2mm, surface_dice_3mm


def main():
    # targeted_folder = "./3d_slicer_checker/pred"
    data = []

    prediction_file_path = "./3d_slicer_checker/pred/s0046.nii.gz"
    sitk_pred = sitk.ReadImage(prediction_file_path)
    sitk_spacing = sitk_pred.GetSpacing()
    array_pred = sitk.GetArrayFromImage(sitk_pred)

    ground_truth_file_path = "./3d_slicer_checker/gt/s0046.nii.gz"
    sitk_gt = sitk.ReadImage(ground_truth_file_path)
    array_gt = sitk.GetArrayFromImage(sitk_gt)

    pred_mask = np.zeros_like(array_pred)
    gt_mask = np.zeros_like(array_gt)

    for i in range(1, 4):
        pred_mask[array_pred == i] = 1
        gt_mask[array_gt == i] = 1
        array_pred_bool = pred_mask.astype(bool)
        array_gt_bool = gt_mask.astype(bool)

    #     (DSC, hausdorff_distance, HD95, average_distance, surface_dice_1mm,
    #      surface_dice_2mm, surface_dice_3mm) = compute_all_metrics(array_pred_bool, array_gt_bool, sitk_spacing)
    #
    #     data.append({"Region_num": i, "DSC": DSC, "hausdorff_distance": hausdorff_distance, "HD95": HD95,
    #                  "average_distance": average_distance, "surface_dice_1mm": surface_dice_1mm,
    #                  "surface_dice_2mm": surface_dice_2mm, "surface_dice_3mm": surface_dice_3mm})
    #
    # csv_file = "./3d_slicer_checker/metrics.csv"
    # # Write the data to the CSV file
    # with open(csv_file, "w", newline="") as csv_file:
    #     fieldnames = ["Region_num", "DSC", "hausdorff_distance", "HD95", "average_distance", "surface_dice_1mm",
    #                   "surface_dice_2mm", "surface_dice_3mm"]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #
    #     writer.writeheader()
    #
    #     for row in data:
    #         writer.writerow(row)

if __name__ == "__main__":
    main()
