import os
import csv
import random

# Specify the directory where you want to search for files
targeted_folder = "../data/three_regions_segmentation_orig/First_60_cases"

# Initialize a list to store file names and splits
data = []
train_ratio, val_ratio, test_ratio = 0.8, 0.2, 0.0
# Iterate through the files in the directory
for root, dirs, files in os.walk(targeted_folder):
    for file in files:
        if file.endswith(".nii.gz"):
            # Randomly assign "train", "val", or "test" with specified ratios
            random_number = random.random()
            if random_number < train_ratio:
                split = "train"
            elif random_number < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"

            data.append({"image_id": file, "split": split})

# Specify the output CSV file
csv_file = "../data/three_regions_segmentation_orig/meta.csv"

# Write the data to the CSV file
with open(csv_file, "w", newline="") as csv_file:
    fieldnames = ["image_id", "split"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for row in data:
        writer.writerow(row)

print("Done!")
