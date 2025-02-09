import os
import xarray as xr

# Define input and output directories
input_dir = "/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/regridded_025/yearly"
output_dir = "/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/regridded_025_renamed/yearly"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all NetCDF files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".nc"):  # Process only .nc files
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name.replace(".nc", "_renamed.nc"))
        
        # Open the NetCDF file
        ds = xr.open_dataset(input_file)
        
        # Rename the coordinates
        ds = ds.rename({"lon": "longitude", "lat": "latitude"})
        
        # Save the updated dataset to the output directory
        ds.to_netcdf(output_file)
        print(f"Processed: {file_name} -> {os.path.basename(output_file)}")

print("All files processed successfully.")