import xarray as xr
import numpy as np

# Load datasets
ds_pm = xr.open_dataset("/l/users/fahad.khan/akhtar/Pangu/data/data_prep/pm_flipped/2019/cams_201901_025_renamed_flipped.nc")
ds_surface = xr.open_dataset("/l/users/fahad.khan/akhtar/Pangu/data/data_prep/surface-orig/2019/surface_201901.nc")

# Flip surface latitudes to match orientation if necessary
if not np.all(ds_pm.latitude.values == ds_surface.latitude.values[::-1]):
    ds_surface = ds_surface.sortby('latitude')

# Match the time resolution: subsample surface data to match PM timestamps (every 3 hours)
common_times = np.intersect1d(ds_pm.time.values, ds_surface.time.values)
ds_pm = ds_pm.sel(time=common_times)
ds_surface = ds_surface.sel(time=common_times)

# Merge datasets (should auto-align by time/lat/lon)
merged = xr.merge([ds_surface, ds_pm])

# Save to new NetCDF file
merged.to_netcdf("/l/users/fahad.khan/akhtar/Pangu/data/data_prep/merged/surface_201901.nc")