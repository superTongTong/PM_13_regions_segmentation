import os
import csv
import random

# Specify the directory where you want to search for files
targeted_folder = "D:/master/graduation_project/data_set/abdomenCT_198_test_cases/images"

# Initialize a list to store file names and splits
data = []
train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2 # for 198 cases, 117 train, 42 val, 39 test
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
csv_file = "D:/master/graduation_project/data_set/abdomenCT_198_test_cases/meta.csv"

# Write the data to the CSV file
with open(csv_file, "w", newline="") as csv_file:
    fieldnames = ["image_id", "split"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for row in data:
        writer.writerow(row)

print("Done!")
