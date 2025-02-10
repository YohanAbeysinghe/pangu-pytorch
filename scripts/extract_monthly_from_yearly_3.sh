#!/bin/bash

# Define the specific file to process
FILE="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2020/monthly/upper_202001.nc"

# Extract the filename (e.g., upper_202001.nc)
FILENAME=$(basename "$FILE")

# Extract the year from the filename (e.g., 2020)
YEAR=$(echo "$FILENAME" | grep -oP '\d{4}')

# Define the output folder for the year
OUTPUT_BASE_FOLDER="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2020/jan"
OUTPUT_FOLDER="$OUTPUT_BASE_FOLDER/$YEAR"

# Create the output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Split the .nc file into monthly files and save them in the year folder
echo "Processing $FILE and saving results to $OUTPUT_FOLDER"
cdo splitmon "$FILE" "$OUTPUT_FOLDER/upper_${YEAR}"

echo "Processing complete. Files are saved in $OUTPUT_FOLDER."