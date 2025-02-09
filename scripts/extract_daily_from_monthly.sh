#!/bin/bash

# Input NetCDF file (containing a whole year of data)
INPUT_FILE="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2009/upper_2009.nc"

# Output directory for monthly files
OUTPUT_DIR="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2009/monthly"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract unique year-month values (YYYY-MM) from the input file
IFS=',' read -ra MONTHS <<< "$(cdo -s showdate "$INPUT_FILE" | cut -c1-7 | sort -u | tr '\n' ',')"

# Loop through each unique month
for MONTH in "${MONTHS[@]}"; do
    # Remove spaces (just in case)
    MONTH=$(echo "$MONTH" | xargs)
    
    # Convert date format from YYYY-MM to YYYYMM
    MONTH_FORMATTED=$(echo "$MONTH" | tr -d '-')

    # Set the output filename
    OUTPUT_FILE="${OUTPUT_DIR}/upper_${MONTH_FORMATTED}.nc"

    echo "Extracting data for ${MONTH} into ${OUTPUT_FILE}..."

    # Extract data for the specific month
    cdo selmon,"$(echo "$MONTH" | cut -d'-' -f2)" "$INPUT_FILE" "$OUTPUT_FILE" || { 
        echo "Error extracting $MONTH. Skipping..." 
        continue 
    }
done

echo "Extraction complete. Monthly files are saved in $OUTPUT_DIR."