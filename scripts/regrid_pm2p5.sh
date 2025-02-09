#!/bin/bash

# Input and output directories
input_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/original_075/yearly"
output_dir="/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/regridded_025/yearly"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all .nc files in the input directory
for input_file in "$input_dir"/*.nc; do
  # Extract the filename without the directory
  base_name=$(basename "$input_file" .nc)
  
  # Construct the output filename with _025 appended
  output_file="$output_dir/${base_name}_025.nc"
  
  # Perform the regridding
  cdo remapbil,r1440x721 "$input_file" "$output_file"
  
  # Check for errors
  if [[ $? -ne 0 ]]; then
    echo "Error processing $input_file"
  else
    echo "Regridded $input_file -> $output_file"
  fi
done