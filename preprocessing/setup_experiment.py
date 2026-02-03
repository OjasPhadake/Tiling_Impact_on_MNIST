import os
import tensorflow
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

BASE_DIR = "../"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
TILED_DIR = os.path.join(BASE_DIR, "data/tiled")
SUBSETS = {'train' : 10000, 'test': 2000}
TILE_SIZE = 7

def create_dirs():
    for path in [RAW_DIR, TILED_DIR]:
        for sub in ['train', 'test']:
            os.makedirs(os.path.join(path, sub), exist_ok = True)

def apply_tiling_hypothesis(img_array, tile_size=7):
    h, w = img_array.shape
    new_img = np.zeros_like(img_array)

    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = img_array[i:i+tile_size, j:j+tile_size].copy()

            mask = np.zeros((tile_size, tile_size))
            mask[1:-1, 1:-1] = 1
            tile = tile * mask

            new_img[i:i+tile_size, j:j+tile_size] = tile

    return new_img

def run_setup():
    create_dirs()

    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

    data_map = {
        'train' : (x_train_full[:10000], y_train_full[:10000]), 
        'test' : (x_test_full[:2000], y_test_full[:2000])
    }

    for phase, (images, labels) in data_map.items():
        print(f"Processing {len(images)} {phase} images...")
        for idx, (img, label) in enumerate(zip(images, labels)):

            raw_path = os.path.join(RAW_DIR, phase, f"{idx}_label_{label}.png")
            Image.fromarray(img).save(raw_path)

            tiled_img = apply_tiling_hypothesis(img, TILE_SIZE)
            tiled_path = os.path.join(TILED_DIR, phase, f"{idx}_label_{label}.png")
            Image.fromarray(tiled_img.astype(np.uint8)).save(tiled_path)


if __name__ == "__main__":
    run_setup()
    print("\nExperiment folders are ready")