from config import setup_nnunet
import subprocess



def plan_and_preprocess():
    # setup the nnUNet environment variables
    setup_nnunet()

    '''Run the following commands for new data, if you want train from scratch:'''
    command = "nnUNetv2_plan_and_preprocess -d 18 -pl ExperimentPlanner -c 3d_fullres -np 2 --verify_dataset_integrity"

    '''Run the following commands for new data, if you want to use pre-trained model, step 1:'''
    # command = ("nnUNetv2_move_plans_between_datasets -s 4 -t 8 "
    #            "-sp plans -tp nnUNetPlans_finetune")

    '''Run the following commands extract fingerprint of new data, if you want to use pre-trained model, step 2:'''
    # command = ("nnUNetv2_extract_fingerprint -d 8")

    '''Run the following commands to processing new data, if you want to use pre-trained model, step 3:'''
    # command = ("nnUNetv2_preprocess -d 8 -plans_name nnUNetPlans_finetune "
    #            "-c 3d_fullres -np 2 --verify_dataset_integrity")

    # Run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    plan_and_preprocess()
