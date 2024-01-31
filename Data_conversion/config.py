import os
from pathlib import Path


def get_nnunet_dir():
    # Get the current working directory where the Python script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    target_folder_name = "data"
    while script_directory != "/":
        if target_folder_name in os.listdir(script_directory):
            root_directory = script_directory
            break
        script_directory = os.path.dirname(script_directory)
    else:
        # Handle the case where the target folder was not found
        root_directory = None

    if root_directory:
        config_dir = os.path.join(root_directory, target_folder_name)
        # print(f"The '{target_folder_name}' folder is located at: {os.path.join(root_directory, target_folder_name)}")
    else:
        print(f"The '{target_folder_name}' folder was not found in the directory tree of the script.")

    return Path(config_dir)


def setup_nnunet():

    config_dir = get_nnunet_dir()
    raw_dir = config_dir / "nnunet/raw"
    preprocessed_dir = config_dir / "nnunet/preprocessed"
    results_dir = config_dir / "nnunet/results"

    # This variables will only be active during the python script execution.
    # Therefore, we do not have to unset them in the end.
    os.environ["nnUNet_raw"] = str(raw_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_results"] = str(results_dir)


def mian():
    setup_nnunet()
    print("nnUNet environment variable is set up.")
    print(os.environ["nnUNet_raw"])
    print(os.environ["nnUNet_preprocessed"])
    print(os.environ["nnUNet_results"])
    # print(get_filenames_of_train_images_and_targets(os.path.join(os.environ["nnUNet_raw"], 'Dataset007_CKI_DATA')))

if __name__ == "__main__":
    mian()
