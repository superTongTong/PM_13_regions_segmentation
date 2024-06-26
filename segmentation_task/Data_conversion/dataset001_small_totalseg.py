import sys
import os
from pathlib import Path
import shutil
import json

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from map_to_binary import class_map_PM
from config import setup_nnunet

def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels):
    print("Creating dataset.json...")
    setup_nnunet()
    out_base = Path(os.environ['nnUNet_raw']) / foldername

    json_dict = {}
    json_dict['name'] = "PM 13 regions segmentation"
    json_dict['description'] = "Segmentation of PM project"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "1.0"
    json_dict['channel_names'] = {"0": "CT"}
    json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + list(labels))}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / foldername
    output_folder_pkl.mkdir(exist_ok=True)

    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)


def combine_labels(ref_img, file_out, masks):
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)  
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx+1
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


if __name__ == "__main__":    
    """
    Convert the dataset to nnUNet format and generate dataset.json and splits_final.json

    example usage: 
    python dataset001_small_totalseg.py C:/Users/20202119/PycharmProjects/segmentation_PM/data/totalseg_small_dataset C:/Users/20202119/PycharmProjects/segmentation_PM/data/nnunet/raw/Dataset001_SmallTotalseg class_map_24_organs

    You must set nnUNet_raw and nnUNet_preprocessed environment variables before running this (see nnUNet documentation).
    To set environment variables in Windows, use the following command in Command Prompt
    set nnUNet_raw=C:/Users/20202119/PycharmProjects/segmentation_PM/data/nnunet/raw
    set nnUNet_preprocessed=C:/Users/20202119/PycharmProjects/segmentation_PM/data/nnunet/preprocessed
    set nnUNet_results=C:/Users/20202119/PycharmProjects/segmentation_PM/data/nnunet/results
    """

    dataset_path = Path(sys.argv[1])  # directory containining all the subjects
    nnunet_path = Path(sys.argv[2])  # directory of the new nnunet dataset
    # nnunet_path.mkdir(parents=True, exist_ok=True)
    # map_to_binary contains 3 list for 3 different dataset. Choose which one you want to produce. Choose from:
    #   class_map_24_organs : 24 classes
    #   class_map_13_regions : 13 classes
    #   class_map_abdomenCT :  4 classes

    class_map_name = sys.argv[3]  

    class_map = class_map_PM[class_map_name]

    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(dataset_path / "meta.csv", sep=";")
    subjects_train = meta[meta["split"] == "train"]["image_id"].to_list()
    subjects_val = meta[meta["split"] == "val"]["image_id"].to_list()
    subjects_test = meta[meta["split"] == "test"]["image_id"].to_list()

    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTr" / f"{subject}.nii.gz",
                       [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    print("Copying test data...")
    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTs" / f"{subject}.nii.gz",
                       [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, class_map.values())

