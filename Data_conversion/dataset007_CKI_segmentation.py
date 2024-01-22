import sys
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm
from map_to_binary import class_map_PM
from dataset001_small_totalseg import generate_json_from_dir_v2
import re


if __name__ == "__main__":
    """
    Convert the dataset to nnUNet format and generate dataset.json and splits_final.json

    example usage: 
    python dataset007_CKI_segmentation.py D:/master/graduation_project/some/where C:/Users/20202119/PycharmProjects/segmentation_PM/data/nnunet/raw/Dataset007_CKI_DATA class_map_3_regions

    You must set nnUNet_raw and nnUNet_preprocessed environment variables before running this (see nnUNet documentation)

    """

    dataset_path = Path(sys.argv[1])  # directory containing all the subjects
    nnunet_path = Path(sys.argv[2])  # directory of the new nnunet dataset
    # nnunet_path.mkdir(parents=True, exist_ok=True)
    '''
       map_to_binary contains 3 list for 3 different dataset. Choose which one you want to produce. 
       Choose from:
       class_map_24_organs : 24 classes
       class_map_13_regions : 13 classes
       class_map_abdomenCT :  4 classes
       class_map_3_regions :  3 classes
    '''
    class_map_name = sys.argv[3]

    class_map = class_map_PM[class_map_name]

    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(dataset_path / "meta.csv", sep=",")
    subjects_train = meta[meta["split"] == "train"]["image_id"].to_list()
    subjects_val = meta[meta["split"] == "val"]["image_id"].to_list()

    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / "images" / subject
        shutil.copy(subject_path, nnunet_path / "imagesTr" / f"{subject}")

        label = re.sub(r'_(\d+)_\d+\.', r'_\1.', subject)
        label_path = dataset_path / "masks" / label
        shutil.copy(label_path, nnunet_path / "labelsTr" / f"{label}")


    # Extract the common part of the filenames for comparison
    common_part_ct_scan = "_0000.nii.gz"
    subjects_train_new = [subject.replace(common_part_ct_scan, "") for subject in subjects_train]
    subjects_val_new = [subject.replace(common_part_ct_scan, "") for subject in subjects_val]
    generate_json_from_dir_v2(nnunet_path.name, subjects_train_new, subjects_val_new, class_map.values())
