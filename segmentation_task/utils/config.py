import os
from pathlib import Path


def get_weights_dir():
    # Get the current working directory where the Python script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    target_folder_name = "weight_path"
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
        print(f"The '{target_folder_name}' folder is located at: {os.path.join(root_directory, target_folder_name)}")
    else:
        print(f"The '{target_folder_name}' folder was not found in the directory tree of the script.")

    return Path(config_dir)




