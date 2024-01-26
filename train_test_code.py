from Data_conversion.config import setup_nnunet
import subprocess
from pathlib import Path


def main():
    setup_nnunet()
    # set the command-line arguments as needed.
    pre_trained_weight = Path("./data/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")
    command = (f'python ./nnunetv2/run/run_training.py 7 3d_fullres 0'
               f' -tr nnUNetTrainer_2epochs -p nnUNetPlans_finetune '
               f'-pretrained_weights {pre_trained_weight}')

    # Run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":

    main()
