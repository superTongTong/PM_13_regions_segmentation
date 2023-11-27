from Data_conversion.config import setup_nnunet
import subprocess
from pathlib import Path



def train():
    # setup the nnUNet environment variables
    setup_nnunet()
    pre_trained_weight = Path("./data/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")
    # set the command-line arguments as needed.
    command = f"nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_5epochs -p nnUNetPlans_pretrain -pretrained_weights {pre_trained_weight}"
    # command = f"nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_5epochs"
    # Run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    train()
