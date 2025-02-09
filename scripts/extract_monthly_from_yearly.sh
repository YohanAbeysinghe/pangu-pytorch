#!/bin/bash

# Input folder containing yearly .nc files
INPUT_FOLDER="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_mena_12hour/yearly"

# Output base folder
OUTPUT_BASE_FOLDER="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_mena_12hour"

# Loop through all .nc files in the input folder
for FILE in "$INPUT_FOLDER"/*.nc; do
    # Extract the filename (e.g., surface_2003.nc)
    FILENAME=$(basename "$FILE")
    
    # Extract the year from the filename (e.g., 2003)
    YEAR=$(echo "$FILENAME" | grep -oP '\d{4}')
    
    # Create the output folder for the year
    OUTPUT_FOLDER="$OUTPUT_BASE_FOLDER/$YEAR"
    mkdir -p "$OUTPUT_FOLDER"
    
    # Split the .nc file into monthly files and save them in the year folder
    echo "Processing $FILE and saving results to $OUTPUT_FOLDER"
    cdo splitmon "$FILE" "$OUTPUT_FOLDER/surface_${YEAR}"
done

echo "Processing complete. Files are saved in year-specific folders under $OUTPUT_BASE_FOLDER."