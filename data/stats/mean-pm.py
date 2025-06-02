import xarray as xr
import numpy as np
import glob
import decimal

# Set precision context for Decimal
decimal.getcontext().prec = 20

# File pattern and output file
input_pattern = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/surface/surface_2019*.nc"
output_path = "/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch/data/stats/pm_stats_2019.txt"

# PM variables
pm_vars = ['pm1', 'pm2p5', 'pm10']

# Open dataset
ds = xr.open_mfdataset(sorted(glob.glob(input_pattern)), combine='by_coords')

# Compute stats using high-precision Decimal
stats = {}
for var in pm_vars:
    data = ds[var].values  # shape: [time, lat, lon]
    flat = data.reshape(-1)  # flatten to 1D
    flat = flat[~np.isnan(flat)]  # remove NaNs

    mean_val = decimal.Decimal(str(np.mean(flat)))
    std_val = decimal.Decimal(str(np.std(flat)))

    stats[var] = {'mean': mean_val, 'std': std_val}

# Write to text file
with open(output_path, 'w') as f:
    f.write("High-Precision PM Statistics for 2019\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Variable':<10} {'Mean':>25} {'Std Dev':>25}\n")
    f.write("-" * 50 + "\n")
    for var in pm_vars:
        f.write(f"{var:<10} {stats[var]['mean']:>25.10e} {stats[var]['std']:>25.10e}\n")

print(f"PM statistics written to {output_path}")