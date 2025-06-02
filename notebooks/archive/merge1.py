import xarray as xr
import numpy as np

# Paths to your input files
pm_path = "/l/users/fahad.khan/akhtar/Pangu/data/data_prep/pm/2019/cams_201901_025_renamed.nc"
surface_path = "/l/users/fahad.khan/akhtar/Pangu/data/data_prep/surface-orig/2019/surface_201901.nc"
output_path = "/l/users/fahad.khan/akhtar/Pangu/data/data_prep/merged/surface_201901.nc"


# Open datasets
ds_pm = xr.open_dataset(pm_path)
ds_surface = xr.open_dataset(surface_path)

# Step 1: Ensure time alignment
common_times = np.intersect1d(ds_surface.time.values, ds_pm.time.values)
ds_pm_aligned = ds_pm.sel(time=common_times)
ds_surface_aligned = ds_surface.sel(time=common_times)

# Step 2: Force both datasets to have latitudes in the same order (e.g., ascending)
if ds_surface_aligned.latitude.values[0] > ds_surface_aligned.latitude.values[-1]:
    ds_surface_aligned = ds_surface_aligned.sortby("latitude")

if ds_pm_aligned.latitude.values[0] > ds_pm_aligned.latitude.values[-1]:
    ds_pm_aligned = ds_pm_aligned.sortby("latitude")

# Step 3: Merge
ds_merged = xr.merge([ds_surface_aligned, ds_pm_aligned])

# Step 4: Save to new NetCDF
ds_merged.to_netcdf(output_path)

print(f"✅ Merged file saved at: {output_path}")