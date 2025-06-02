import xarray as xr
import numpy as np
import glob

# File paths
input_pattern = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/surface/surface_2019*.nc"
output_path = "/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch/data/stats/surface_stats_2019.txt"

# Variables of interest
variables = ['u10', 'v10', 't2m', 'msl', 'pm1', 'pm2p5', 'pm10']

# Open all 2019 files as one dataset
ds = xr.open_mfdataset(sorted(glob.glob(input_pattern)), combine='by_coords')

# Compute statistics
stats = {}
for var in variables:
    data = ds[var]
    mean_val = float(data.mean(dim='time').values.mean())  # Global average
    std_val = float(data.std(dim='time').values.std())     # Global std
    stats[var] = {'mean': mean_val, 'std': std_val}

# Write to file
with open(output_path, 'w') as f:
    f.write("Surface Variable Statistics for 2019\n")
    f.write("=" * 40 + "\n")
    f.write(f"{'Variable':<10} {'Mean':>15} {'Std Dev':>15}\n")
    f.write("-" * 40 + "\n")
    for var in variables:
        f.write(f"{var:<10} {stats[var]['mean']:>15.4f} {stats[var]['std']:>15.4f}\n")

print(f"Statistics written to {output_path}")