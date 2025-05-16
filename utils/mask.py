import numpy as np

def generate_spatial_mask(lat_grid, lon_grid, valid_latlon_pairs):
    """
    Generate a binary spatial mask where entries corresponding to
    valid (lat, lon) pairs are 1, others are 0.

    Args:
        lat_grid (list or np.ndarray): 1D array of latitude values
        lon_grid (list or np.ndarray): 1D array of longitude values
        valid_latlon_pairs (list of tuples): List of (lat, lon) coordinates to include in mask

    Returns:
        mask: np.ndarray of shape [H, W] with 1s at valid points, 0 elsewhere
    """
    lat_idx = {lat: i for i, lat in enumerate(lat_grid)}
    lon_idx = {lon: i for i, lon in enumerate(lon_grid)}

    H = len(lat_grid)
    W = len(lon_grid)
    mask = np.zeros((H, W), dtype=np.float32)

    for lat, lon in valid_latlon_pairs:
        if lat in lat_idx and lon in lon_idx:
            h = lat_idx[lat]
            w = lon_idx[lon]
            mask[h, w] = 1.0

    return mask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_mask_vs_dataframe(mask, lat_grid, lon_grid, df,
                                 lat_key='lat', lon_key='lon', title='Mask vs Data Presence'):
    """
    Visualize the spatial mask and actual data points from a DataFrame in separate subplots.

    Args:
        mask (np.ndarray): shape [H, W], binary mask.
        lat_grid (list or np.ndarray): latitude grid values.
        lon_grid (list or np.ndarray): longitude grid values.
        df (pd.DataFrame): DataFrame with 'lat' and 'lon' columns.
        lat_key (str): name of the latitude column in df.
        lon_key (str): name of the longitude column in df.
        title (str): plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot mask only
    axes[0].imshow(mask, cmap='Greys', extent=[lon_grid[0], lon_grid[-1], lat_grid[0], lat_grid[-1]],
                   origin='lower', aspect='auto')
    axes[0].set_title("Mask Grid")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True)

    # Plot data points only
    axes[1].scatter(df[lon_key], df[lat_key], s=10, color='red', alpha=0.7)
    axes[1].set_xlim([lon_grid[0], lon_grid[-1]])
    axes[1].set_ylim([lat_grid[0], lat_grid[-1]])
    axes[1].set_title("Data Points")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def count_mask_hits(df, lat_grid, lon_grid, mask):
    lat_to_idx = {lat: i for i, lat in enumerate(lat_grid)}
    lon_to_idx = {lon: i for i, lon in enumerate(lon_grid)}

    hits = 0
    total = 0

    for row in df.itertuples(index=False):
        lat, lon = getattr(row, 'lat'), getattr(row, 'lon')
        if lat in lat_to_idx and lon in lon_to_idx:
            h, w = lat_to_idx[lat], lon_to_idx[lon]
            if mask[h, w] == 1.0:
                hits += 1
        total += 1

    return hits, total, hits == total
def count_mask_utilization(mask, valid_pairs, lat_grid, lon_grid):
    used_mask = np.zeros_like(mask)

    lat_to_idx = {lat: i for i, lat in enumerate(lat_grid)}
    lon_to_idx = {lon: i for i, lon in enumerate(lon_grid)}

    for lat, lon in valid_pairs:
        if lat in lat_to_idx and lon in lon_to_idx:
            h = lat_to_idx[lat]
            w = lon_to_idx[lon]
            used_mask[h, w] = 1

    unused = np.logical_and(mask == 1, used_mask == 0)
    return np.sum(unused), unused
