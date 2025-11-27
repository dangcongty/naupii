import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def generate_heatmap(points, width, height, sigma):
    """
    Generate a heatmap by placing 1s at points and applying Gaussian filter.
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            heatmap[int(y), int(x)] = 1.0
    if heatmap.sum() > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    else:
        heatmap = np.zeros((height, width), dtype=np.float32)
    return heatmap

if __name__ == "__main__":
    with open('datasets/data.json', 'r') as f:
        data = json.load(f)

    heatmap_dir = 'datasets/heatmap'
    os.makedirs(heatmap_dir, exist_ok=True)

    width = 2813
    height = 2121
    sigma = 5

    for path, points in tqdm(data.items()):
        heatmap = generate_heatmap(points, width, height, sigma)
        np.save(f'{heatmap_dir}/{os.path.basename(path)[:-4]}.npy', heatmap)
