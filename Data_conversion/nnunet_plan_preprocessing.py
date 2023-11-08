from config import setup_nnunet
import subprocess


def plan_and_preprocess():
    # setup the nnUNet environment variables
    setup_nnunet()

    # set the command-line arguments as needed.
    command = "nnUNetv2_plan_and_preprocess -d 2 -pl ExperimentPlanner -c 3d_fullres -np 2 --verify_dataset_integrity"

    # Run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    plan_and_preprocess()
