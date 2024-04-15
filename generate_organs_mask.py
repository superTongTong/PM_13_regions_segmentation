import argparse
from pathlib import Path
import os
import subprocess
import time


def generate_organs_mask(input_folder, output_path):

    # set the command-line arguments as needed.
    command = (f'python ./totalsegmentator/TotalSegmentator.py -i {input_folder} -o {output_path} '
               f'--roi_subset stomach pancreas liver spleen')

    # Run the command
    subprocess.run(command, shell=True)

def process_folders(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    for file in os.listdir(input_folder):
        if file.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, file)
            save_dir = os.path.join(output_folder, file[:5])
            # Create the output folder
            os.makedirs(output_folder, exist_ok=True)
            generate_organs_mask(input_path, save_dir)
            print(f"Processed data have been saved to '{save_dir}'.")

def main(segmentations=None):

    parser = argparse.ArgumentParser(description="generate liver, stomach, pancreas, spleen masks.")

    parser.add_argument("-i", '--input',
                        default=r".\code_test_folder\for_evaluation\images",
                        help="CT nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute())

    parser.add_argument("-o", '--output',
                        default=r".\code_test_folder\for_evaluation\organs_masks",
                        help="Output directory for segmentation masks",
                        type=lambda p: Path(p).absolute())

    args = parser.parse_args()
    process_folders(args.input, args.output)


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = round(time.time() - start_time, 2)
    print(f"--- {total_time} seconds ---") # process 10 cases takes 10 minutes
