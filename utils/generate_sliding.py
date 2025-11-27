import math
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def sliding_windows(img, heatmap=None, win_size=640, step=320):
    """
    Slide a window over an image (and optional heatmap) with padding.

    Returns:
        windows: list of image crops
        heatmaps: list of heatmap crops (if heatmap given)
        coords: list of top-left coordinates (x, y)
        img_padded: padded image
        heatmap_padded: padded heatmap
    """
    h, w = img.shape[:2]

    # --- Padding để chia hết ---
    pad_h = (math.ceil(h / step) * step) - h
    pad_w = (math.ceil(w / step) * step) - w

    img_padded = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )

    heatmap_padded = None
    if heatmap is not None:
        heatmap_padded = np.pad(
            heatmap,
            ((0, pad_h), (0, pad_w)),
            mode='constant', constant_values=0
        )

    padded_h, padded_w = img_padded.shape[:2]

    windows = []
    heatmaps_list = [] if heatmap is not None else None
    coords = []

    # --- Trượt cửa sổ ---
    for y in range(0, padded_h - win_size + 1, step):
        for x in range(0, padded_w - win_size + 1, step):
            windows.append(img_padded[y:y+win_size, x:x+win_size])
            if heatmap is not None:
                heatmaps_list.append(heatmap_padded[y:y+win_size, x:x+win_size])
            coords.append((x, y))

    return windows, heatmaps_list, coords, img_padded, heatmap_padded


if __name__ == "__main__":
    image_paths = 'datasets/images'
    save_path_img = 'datasets/slide/images/'
    save_path_hm = 'datasets/slide/heatmaps/'
    os.makedirs(save_path_img, exist_ok=True)
    os.makedirs(save_path_hm, exist_ok=True)

    sliding_size = 640
    step = 320

    for path in tqdm(glob(f'{image_paths}/*')):
        img = cv2.imread(path)
        heatmap = np.load(f'datasets/heatmap/{os.path.basename(path)[:-4]}.npy')

        windows, heatmaps_list, coords, _, _ = sliding_windows(img, heatmap, sliding_size, step)

        for i, (win, coord) in enumerate(zip(windows, coords)):
            x, y = coord
            cv2.imwrite(f"{save_path_img}/{os.path.basename(path)[:-4]}_{i}_{x}_{y}.jpg", win)
            if heatmaps_list is not None:
                np.save(f"{save_path_hm}/{os.path.basename(path)[:-4]}_{i}_{x}_{y}.npy", heatmaps_list[i])
