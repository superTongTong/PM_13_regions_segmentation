import os
import csv
import random

# Specify the directory where you want to search for files
targeted_folder = "C:/Users/20202119/PycharmProjects/segmentation_PM/data/abdomenCT_60_test_cases/images"

# Initialize a list to store file names and splits
data = []

# Iterate through the files in the directory
for root, dirs, files in os.walk(targeted_folder):
    for file in files:
        if file.endswith(".nii.gz"):
            # Randomly assign "train" or "val" with a 95% / 5% split
            split = "train" if random.random() < 0.95 else "val"
            data.append({"image_id": file, "split": split})

# Specify the output CSV file
csv_file = "../data/abdomenCT_60_test_cases/meta.csv"

# Write the data to the CSV file
with open(csv_file, "w", newline="") as csv_file:
    fieldnames = ["image_id", "split"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for row in data:
        writer.writerow(row)

print("Done!")
