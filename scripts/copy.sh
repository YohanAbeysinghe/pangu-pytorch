#!/bin/bash

# Source directory
SOURCE_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour"

# Destination directory (replace with your actual destination path)
DEST_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/non_cropped_model/upper"

# Loop through the year folders (2003, 2004, ..., 2023)
for year in $(ls "$SOURCE_DIR"); do
  # Make sure we're only dealing with directories (skip non-directories)
  if [ -d "$SOURCE_DIR/$year" ]; then
    echo "Processing files from $year..."

    # Use find to locate all files in the year folder (ignore subfolders like 'monthly')
    find "$SOURCE_DIR/$year" -mindepth 1 -maxdepth 1 -type f -exec cp {} "$DEST_DIR" \;
  fi
done

echo "File copy completed!"