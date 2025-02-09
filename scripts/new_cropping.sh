#!/bin/bash

# Directory where the files are located
input_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_12hour/2005/daily"
output_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/upper/upper_mena_12hour/2005"
# input_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/regridded_025_renamed/yearly"
# output_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/cropped_regridded_025_renamed/yearly"
# input_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_12hour/yearly"
# output_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_mena_12hour/yearly"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Function to crop a file
crop_file() {
    file="$1"
    base_name=$(basename "$file")
    output_file="$output_dir/$base_name"
    echo "Cropping $base_name..."
    cdo sellonlatbox,-0.01,76.01,-7.01,45.01 "$file" "$output_file" && echo "Saved cropped file to $output_file" || echo "Failed to crop $base_name"
}

export -f crop_file

# Check if GNU parallel is installed
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for faster processing..."
    # Find files and process them in parallel
    find "$input_dir" -name "*.nc" | parallel -j "$(nproc)" crop_file
else
    echo "GNU parallel not found. Processing files sequentially..."
    # Sequential processing if parallel is not available
    for file in "$input_dir"/*.nc; do
        crop_file "$file"
    done
fi