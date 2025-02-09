#!/bin/bash

# Input directory containing monthly NetCDF files
INPUT_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2005/monthly"

# Output directory for daily files
OUTPUT_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2005/daily"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each NetCDF file in the input directory
for INPUT_FILE in "$INPUT_DIR"/*.nc; do
    # Get the base filename (e.g., upper_201112.nc)
    BASENAME=$(basename "$INPUT_FILE")
    
    echo "Processing file: $INPUT_FILE"

    # Get all unique dates from the input file
    DATES=$(cdo -s showdate "$INPUT_FILE" | tr ',' '\n' | uniq)

    # Loop through each date and extract the corresponding data
    for DATE in $DATES; do
        # Convert date format from YYYY-MM-DD to YYYYMMDD
        DATE_FORMATTED=$(echo $DATE | tr -d '-')
        
        # Set the output filename
        OUTPUT_FILE="${OUTPUT_DIR}/upper_${DATE_FORMATTED}.nc"
        
        echo "  Extracting data for ${DATE} into ${OUTPUT_FILE}..."
        cdo seldate,${DATE} "$INPUT_FILE" "$OUTPUT_FILE"
    done

    echo "Finished processing file: $INPUT_FILE"
done

echo "All files processed. Daily files are saved in $OUTPUT_DIR."