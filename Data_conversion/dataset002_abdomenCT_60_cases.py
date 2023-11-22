import sys
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm
from map_to_binary import class_map_PM
from dataset001_small_totalseg import generate_json_from_dir_v2
import re
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

if __name__ == "__main__":
    """
    Convert the dataset to nnUNet format and generate dataset.json and splits_final.json

    example usage: 
    python dataset001_small_totalseg.py C:/Users/20202119/PycharmProjects/segmentation_PM/data/abdomenCT_60_test_cases C:/Users/20202119/PycharmProjects/segmentation_PM/data/nnunet/raw/Dataset002_abdomenCT_60 class_map_abdomenCT

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

    meta = pd.read_csv(dataset_path / "meta.csv", sep=",")
    subjects_train = meta[meta["split"] == "train"]["image_id"].to_list()
    subjects_val = meta[meta["split"] == "val"]["image_id"].to_list()
    subjects_test = meta[meta["split"] == "test"]["image_id"].to_list()

    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / "images" / subject
        shutil.copy(subject_path, nnunet_path / "imagesTr" / f"{subject}")

        label = re.sub(r'_(\d+)_\d+\.', r'_\1.', subject)
        label_path = dataset_path / "masks" / label
        shutil.copy(label_path, nnunet_path / "labelsTr" / f"{label}")
        # combine_labels(subject_path / "ct.nii.gz",
        #                nnunet_path / "labelsTr" / f"{subject}.nii.gz",
        #                [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    # print("Copying test data...")
    # for subject in tqdm(subjects_test):
    #     subject_path = dataset_path / subject
    #     shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        # combine_labels(subject_path / "ct.nii.gz",
        #                nnunet_path / "labelsTs" / f"{subject}.nii.gz",
        #                [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, class_map.values())
