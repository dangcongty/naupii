import os
import random
import shutil
from glob import glob


def split_train_val(image_dir='datasets/slide/images',
                    heatmap_dir='datasets/slide/heatmaps',
                    train_ratio=0.8,
                    save_dir='datasets'):
    """
    Split images and heatmaps into train/val and save into folders:
        datasets/train/images
        datasets/train/heatmaps
        datasets/val/images
        datasets/val/heatmaps
    """
    # Create folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(save_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, split, 'heatmaps'), exist_ok=True)

    # Get all images
    image_paths = glob(os.path.join(image_dir, '*.jpg'))
    random.shuffle(image_paths)

    n_train = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:]

    for paths, split in [(train_paths, 'train'), (val_paths, 'val')]:
        for img_path in paths:
            fname = os.path.basename(img_path)
            # Copy image
            shutil.copy(img_path, os.path.join(save_dir, split, 'images', fname))
            # Copy corresponding heatmap
            hm_path = os.path.join(heatmap_dir, fname.replace('.jpg', '.npy'))
            if os.path.exists(hm_path):
                shutil.copy(hm_path, os.path.join(save_dir, split, 'heatmaps', os.path.basename(hm_path)))
            else:
                print(f"[Warning] Heatmap not found for {fname}")

    print(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")


split_train_val(
    image_dir='datasets/slide/images',
    heatmap_dir='datasets/slide/heatmaps',
    train_ratio=0.8,
    save_dir='datasets'
)
