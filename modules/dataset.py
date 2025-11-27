import os
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# def resize(img, heatmap, size=1280):
#     img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
#     heatmap_resized = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_LINEAR)
#     return img_resized, heatmap_resized


# def random_affine(img, heatmap, max_rotate=10, max_translate=0.1, max_scale=0.1):
#     """
#     Apply random rotation, translation, scale
#     """
#     h, w = img.shape[:2]

#     # Random parameters
#     angle = random.uniform(-max_rotate, max_rotate)
#     scale = 1 + random.uniform(-max_scale, max_scale)
#     tx = random.uniform(-max_translate, max_translate) * w
#     ty = random.uniform(-max_translate, max_translate) * h

#     M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
#     M[:, 2] += [tx, ty]

#     img_aug = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#     heatmap_aug = cv2.warpAffine(heatmap, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#     return img_aug, heatmap_aug


# def mosaic_4(images, heatmaps, size=1280):
#     """
#     Combine 4 images and heatmaps into one mosaic
#     """
#     s = size // 2
#     mosaic_img = np.zeros((size, size, 3), dtype=np.uint8)
#     mosaic_hm = np.zeros((size, size), dtype=np.float32)

#     # Top-left, top-right, bottom-left, bottom-right
#     for i, (img, hm) in enumerate(zip(images, heatmaps)):
#         img_resized, hm_resized = resize(img, hm, size=s)
#         if i == 0:
#             mosaic_img[0:s, 0:s] = img_resized
#             mosaic_hm[0:s, 0:s] = hm_resized
#         elif i == 1:
#             mosaic_img[0:s, s:size] = img_resized
#             mosaic_hm[0:s, s:size] = hm_resized
#         elif i == 2:
#             mosaic_img[s:size, 0:s] = img_resized
#             mosaic_hm[s:size, 0:s] = hm_resized
#         else:
#             mosaic_img[s:size, s:size] = img_resized
#             mosaic_hm[s:size, s:size] = hm_resized

#     return mosaic_img, mosaic_hm


# class NaupiiDataset(Dataset):
#     def __init__(self, image_paths, mode='train'):
#         """
#         image_paths: list of image file paths
#         heatmap_paths: list of heatmap .npy paths
#         mode: 'train' or 'val'
#         """
#         assert mode in ['train', 'val']
#         self.image_paths = glob(f'{image_paths}/*jpg')
#         self.mode = mode

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         hm_path = f'datasets/slide/heatmaps/{os.path.basename(img_path)[:-4]}.npy'
#         img = cv2.imread(img_path)
#         hm = np.load(hm_path)

#         # Resize both to 1280
#         img, hm = resize(img, hm, size=1280)

#         if self.mode == 'train':
#             # Apply random affine
#             img, hm = random_affine(img, hm)

#             # Mosaic with 3 other random images
#             indices = [index] + random.choices(range(len(self.image_paths)), k=3)
#             imgs = []
#             hms = []
#             for idx in indices:
#                 mosaic_img_path = self.image_paths[idx]
#                 mosaic_hm_path = f'datasets/slide/heatmaps/{os.path.basename(mosaic_img_path)[:-4]}.npy'
#                 tmp_img = cv2.imread(mosaic_img_path)
#                 tmp_hm = np.load(mosaic_hm_path)
#                 imgs.append(tmp_img)
#                 hms.append(tmp_hm)
#             img, hm = mosaic_4(imgs, hms, size=1280)

#         # Convert to tensor
#         img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
#         hm = torch.from_numpy(hm).unsqueeze(0).float()  # add channel dim

#         return img, hm



def resize(img, heatmap, size):
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_LINEAR)
    return img_resized, heatmap_resized


def random_affine(img, heatmap, max_rotate=10, max_translate=0.1, max_scale=0.1):
    h, w = img.shape[:2]

    # Random parameters
    angle = random.uniform(-max_rotate, max_rotate)
    scale = 1 + random.uniform(-max_scale, max_scale)
    tx = random.uniform(-max_translate, max_translate) * w
    ty = random.uniform(-max_translate, max_translate) * h

    # Affine matrix
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[:, 2] += [tx, ty]

    img_aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    heatmap_aug = cv2.warpAffine(heatmap, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return img_aug, heatmap_aug


def mosaic_4(img_list, hm_list, final_size=1280):
    """Combine 4 images + heatmaps into a single mosaic"""
    s = final_size // 2
    mosaic_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    mosaic_hm = np.zeros((final_size, final_size), dtype=np.float32)

    for i, (img, hm) in enumerate(zip(img_list, hm_list)):
        img_resized, hm_resized = resize(img, hm, s)
        if i == 0:
            mosaic_img[0:s, 0:s] = img_resized
            mosaic_hm[0:s, 0:s] = hm_resized
        elif i == 1:
            mosaic_img[0:s, s:final_size] = img_resized
            mosaic_hm[0:s, s:final_size] = hm_resized
        elif i == 2:
            mosaic_img[s:final_size, 0:s] = img_resized
            mosaic_hm[s:final_size, 0:s] = hm_resized
        else:
            mosaic_img[s:final_size, s:final_size] = img_resized
            mosaic_hm[s:final_size, s:final_size] = hm_resized

    return mosaic_img, mosaic_hm


class NaupiiDataset(Dataset):
    def __init__(self, image_paths, mode='train', size=1280):
        """
        image_paths: list of image files
        heatmap_paths: list of heatmap .npy files
        mode: 'train' or 'val'
        size: final image size
        """
        assert mode in ['train', 'val']
        self.image_paths = glob(f'{image_paths}/*.jpg')
        self.mode = mode
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.mode == 'train':
            # --- Mosaic 4 images ---
            indices = [index]
            others = list(range(len(self.image_paths)))
            others.remove(index)
            random.shuffle(others)
            while len(indices) < 4:
                if others:
                    indices.append(others.pop(0))
                else:
                    indices.append(index)

            imgs = []
            hms = []
            for idx in indices:
                img = cv2.imread(self.image_paths[idx])
                hm = np.load(f'datasets/slide/heatmaps/{os.path.basename(self.image_paths[idx])[:-4]}.npy')

                # Resize BEFORE augment
                img, hm = resize(img, hm, self.size)

                # Apply affine
                img, hm = random_affine(img, hm)
                imgs.append(img)
                hms.append(hm)

            img, hm = mosaic_4(imgs, hms, self.size)

        else:  # val
            img = cv2.imread(self.image_paths[index])
            hm = np.load(f'datasets/slide/heatmaps/{os.path.basename(self.image_paths[index])[:-4]}.npy')
            img, hm = resize(img, hm, self.size)

        # --- Downscale heatmap to half size while keeping sum ---
        hm_sum = hm.sum()
        hm = cv2.resize(hm, (self.size//2, self.size//2), interpolation=cv2.INTER_LINEAR)
        if hm.sum() > 0:
            hm = hm_sum * (hm / hm.sum())

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        hm = torch.from_numpy(hm).unsqueeze(0).float()
        return img, hm

def visualize(img_tensor, heatmap_tensor, alpha=0.5, save_path=None):
    """
    Visualize an image and its heatmap overlay.

    img_tensor   : torch tensor [3,H,W] float in 0-1
    heatmap_tensor: torch tensor [1,H,W] float
    alpha        : overlay transparency
    save_path    : optional path to save image
    """
    # Convert to numpy
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # HWC, 0-1
    img = (img * 255).astype(np.uint8)

    heatmap = heatmap_tensor.squeeze(0).cpu().numpy()  # H,W
    heatmap_norm = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)

    # Display using OpenCV
    cv2.imwrite("test.jpg", overlay)

    # Optional save
    if save_path is not None:
        cv2.imwrite(save_path, overlay)

if __name__ == '__main__':
    
    image_paths = 'datasets/slide/images'
    heatmap_paths = 'datasets/slide/heatmaps'
    dataset = NaupiiDataset(image_paths, mode='train')
    # Visualize one sample
    img, hm = dataset[np.random.randint(0, 150)]
    visualize(img, hm)
