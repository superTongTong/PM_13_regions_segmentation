from Data_conversion.config import setup_nnunet
import subprocess
from pathlib import Path


def train():
    # setup the nnUNet environment variables
    setup_nnunet()
    # pre_trained_weight = Path("./data/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")
    pre_trained_weight = Path("./data/nnunet/results/Dataset007_CKI_DATA/50epochs_2rd/fold_0/checkpoint_best.pth")
    # set the command-line arguments as needed. epochs option: 1,5,10,20,50,100,250,2000,4000,8000
    command = f"nnUNetv2_train 7 3d_fullres 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans_finetune -pretrained_weights {pre_trained_weight}"
    # command = f"nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_5epochs"
    # Run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    train()
