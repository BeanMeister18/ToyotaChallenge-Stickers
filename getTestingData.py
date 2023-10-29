import os
import random
import shutil

# Define the source and destination directories
source_directory = 'training/stickers'  # Replace with the path to your source directory
destination_directory = 'testing/stickers'  # Replace with the path to your destination directory

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# List all files in the source directory
files = os.listdir(source_directory)

# Calculate 10% of the total number of files
num_files_to_move = int(len(files) * 0.1)

# Randomly select 10% of the files
files_to_move = random.sample(files, num_files_to_move)

# Move the selected files to the destination directory
for file_name in files_to_move:
    source_file = os.path.join(source_directory, file_name)
    destination_file = os.path.join(destination_directory, file_name)
    shutil.move(source_file, destination_file)

print(f"{num_files_to_move} files moved from {source_directory} to {destination_directory}.")
