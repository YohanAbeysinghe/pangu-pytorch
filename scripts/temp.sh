#!/bin/bash

# Source directory containing year folders and the yearly folder
SOURCE_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_mena_12hour_withpm2p5"

# Target directory to copy the .nc files
TARGET_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/cropped_without_pm2.5_model/surface"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Loop through all subfolders except the "yearly" folder
for FOLDER in "$SOURCE_DIR"/*; do
    if [[ -d "$FOLDER" && "$(basename "$FOLDER")" != "yearly" ]]; then
        echo "Copying .nc files from $FOLDER to $TARGET_DIR..."
        cp "$FOLDER"/*.nc "$TARGET_DIR"
    fi
done

echo "Copying complete. All .nc files are in $TARGET_DIR."