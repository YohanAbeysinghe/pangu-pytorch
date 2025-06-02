import xarray as xr

# Input and output paths
input_path = "/l/users/fahad.khan/akhtar/Pangu/data/data_prep/pm/2019/cams_201901_025_renamed.nc"
output_path = "/l/users/fahad.khan/akhtar/Pangu/data/data_prep/pm/2019/cams_201901_025_renamed_flipped.nc"

# Open dataset
ds = xr.open_dataset(input_path)

# Flip latitude if needed (from ascending to descending)
if ds.latitude.values[0] < ds.latitude.values[-1]:
    ds = ds.sortby("latitude", ascending=False)

# Save new dataset
ds.to_netcdf(output_path)
print(f"✅ Saved flipped dataset to: {output_path}")