import os
import xarray as xr
from datetime import datetime

# Set paths to the folders containing the datasets
surface_folder = '/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_12hour/yearly'
cams_folder = '/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/pm25/regridded_025_renamed/yearly'
output_folder = '/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-data/data_prep/surface/surface_12hour_withpm2p5/yearly'

# Get list of surface files (surface_yyyymm.nc)
surface_files = [f for f in os.listdir(surface_folder) if f.startswith('surface_') and f.endswith('.nc')]

# Loop over all surface files and merge with corresponding cams files
for surface_file in surface_files:
    # Extract the year and month from the surface file name (e.g., surface_201801.nc -> 201801)
    year = surface_file.split('_')[1][:4]
    
    # Define the corresponding cams file name
    cams_file = f'cams_{year}_025_renamed.nc'
    
    # Define full paths for the surface and cams files
    surface_file_path = os.path.join(surface_folder, surface_file)
    cams_file_path = os.path.join(cams_folder, cams_file)
    
    # Check if the cams file exists
    if os.path.exists(cams_file_path):
        # Load the datasets
        dataset1 = xr.open_dataset(cams_file_path)  # Dataset with pm2p5
        dataset2 = xr.open_dataset(surface_file_path)  # Dataset with u10, v10, t2m, msl
        
        # Merge the datasets
        merged_dataset = xr.merge([dataset2, dataset1])
        
        # Define the output file path
        output_file_path = os.path.join(output_folder, f'surface_{year}.nc')
        
        # Save the merged dataset
        merged_dataset.to_netcdf(output_file_path)
        print(f'Merged {surface_file} with {cams_file} and saved to {output_file_path}')
    else:
        print(f'Corresponding CAMS file for {surface_file} not found: {cams_file_path}')