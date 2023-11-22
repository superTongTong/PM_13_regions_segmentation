from config import setup_nnunet
import subprocess


def plan_and_preprocess():
    # setup the nnUNet environment variables
    setup_nnunet()

    # set the command-line arguments as needed.
    # command = "nnUNetv2_plan_and_preprocess -d 1 -pl ExperimentPlanner -c 3d_fullres -np 2 --verify_dataset_integrity"
    # command = ("nnUNetv2_move_plans_between_datasets -s 4 -t 2 "
    #            "-sp plans -tp nnUNetPlans_pretrain")
    command = "nnUNetv2_preprocess -d 2 -plans_name nnUNetPlans_pretrain"
    # Run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    plan_and_preprocess()
