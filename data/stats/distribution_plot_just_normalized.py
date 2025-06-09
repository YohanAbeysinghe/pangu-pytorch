import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# File pattern
input_pattern = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/surface/surface_2019*.nc"

# PM variables
pm_vars = ['pm1', 'pm2p5', 'pm10']

# Open dataset
ds = xr.open_mfdataset(sorted(glob.glob(input_pattern)), combine='by_coords')

plt.figure(figsize=(12, 4))

# Output directory to save plot
output_dir = "/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch/data/stats/surface_distributions_before_log_transform/For_2019"
os.makedirs(output_dir, exist_ok=True)

for i, var in enumerate(pm_vars):
    data = ds[var].values
    flat = data.reshape(-1)
    flat = flat[~np.isnan(flat)]
    
    # Normalize
    mean_val = np.mean(flat)
    std_val = np.std(flat)
    normalized = (flat - mean_val) / std_val
    
    plt.subplot(1, 3, i + 1)
    plt.hist(normalized, bins=100, color='skyblue', edgecolor='black')
    plt.title(f"Normalized Distribution of {var} (2019)")
    plt.xlabel("Normalized value")
    plt.ylabel("Frequency")

plt.tight_layout()

# Save the figure
plot_path = os.path.join(output_dir, "normalized_pm_distributions_2019.png")
plt.savefig(plot_path)
plt.show()