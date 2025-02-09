#!/bin/bash

# Input directory containing yearly NetCDF files
INPUT_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_12hour/yearly"

# Output directory for monthly files
OUTPUT_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/non_cropped_model/surface"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all NetCDF files in the input directory
for INPUT_FILE in "$INPUT_DIR"/*.nc; do
    # Extract the filename without extension
    FILENAME=$(basename "$INPUT_FILE" .nc)

    # Extract year from filename (assuming format like surface_YYYY.nc)
    YEAR=$(echo "$FILENAME" | grep -oE '[0-9]{4}')

    # Skip files without a valid year
    if [[ -z "$YEAR" ]]; then
        echo "Skipping file $FILENAME (no valid year found)"
        continue
    fi

    echo "Processing file: $INPUT_FILE (Year: $YEAR)"

    # Loop through months 1 to 12
    for MONTH in {1..12}; do
        # Format month as two-digit (01, 02, ..., 12)
        MONTH_FORMATTED=$(printf "%02d" $MONTH)

        # Set the output filename
        OUTPUT_FILE="${OUTPUT_DIR}/surface_${YEAR}${MONTH_FORMATTED}.nc"

        echo "Extracting data for ${YEAR}-${MONTH_FORMATTED} into ${OUTPUT_FILE}..."

        # Extract data for the specific month
        cdo selmon,$MONTH "$INPUT_FILE" "$OUTPUT_FILE" || { 
            echo "Error extracting month $MONTH from $INPUT_FILE. Skipping..." 
            continue 
        }
    done
done

echo "Extraction complete. Monthly files are saved in $OUTPUT_DIR."