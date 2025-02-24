import torch
import numpy as np
    
    
def chem_transform(inp):
    x = torch.from_numpy(inp)
    x = (np.log(torch.max(x, torch.tensor(1e-4))) - np.log(1e-4)) / np.log(1e-4)
    return x.numpy()


def compute_norm(data):
    """
    This is a special function to compute the spatial norm as in the aurora paper. 
    We compute the average of the sum of max values in each timestep (time step is the channel). 
    We then divide this by 2
    Data is of shape (timestep, 1, 32, 64)
    """
    mean = 0
    std = np.mean(np.max(data, axis=(2, 3)))/2
    print(f'mean: {mean}')
    print(f'std: {std}')
    return (data - mean) / std


def compute_freq_bins(data, channel_name, max_clip=None, transform=False):

    if transform:
        print('applying chemical transformations to the data')
        data = compute_norm(data)
        print(f'after normalization, data min: {data.min()}')
        print(f'after normalization, data max: {data.max()}')
        data = chem_transform(data)
        print(f'after transformation, data min: {data.min()}')
        print(f'after transformation, data max: {data.max()}')

    else:
        # for the raw data we only clip the values to remove negative values
        data = np.clip(data, 0, max_clip)

    data = data.astype(np.float16)

    # we need to get the region info for the MENA region
    ddeg_out = 5.625
    lat = np.arange(-90+ddeg_out/2, 90, ddeg_out)
    lon = np.arange(0, 360, ddeg_out)
    region_info  = data_utils.get_region_info('MENAreg', lat, lon, patch_size=2)
    data = data[:, :, region_info['min_h']:region_info['max_h']+1, region_info['min_w']:region_info['max_w']+1]

    # we now compute the histogram bins
    bin_edges = np.histogram_bin_edges(data, bins='auto')

    print('number of bins: ', len(bin_edges))
    bin_edges = bin_edges.tolist()

    # now we compute the frequency of each bin
    freq_counts, bins = np.histogram(data, bins=bin_edges)
    freq_counts = freq_counts.tolist()

    return bin_edges, freq_counts 