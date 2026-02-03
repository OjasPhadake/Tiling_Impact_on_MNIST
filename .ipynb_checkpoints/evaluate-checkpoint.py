import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import json
from models.mlp_baseline import MNIST_MLP

class LocalFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(os.path.join(folder_path, "*.png"))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('L')
        img_np = np.array(image, dtype = np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        label = int(img_path.split('_label_')[-1].split('.')[0])

        return img_tensor, label

def evaluate_model(mode):
    test_path = f"data/{mode}/test"
    model_path = f"results/mlp_{mode}.pth"
    
    if not os.path.exists(model_path):
        print(f"Skipping {mode}: Model file not found.")
        return

    test_loader = DataLoader(LocalFolderDataset(test_path), batch_size=32)
    model = MNIST_MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy for {mode.upper()} model: {accuracy:.2f}%")

def get_predictions(mode, images):
    model = MNIST_MLP()
    model.load_state_dict(torch.load(f"results/mlp_{mode}.pth"))
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    return predicted
    
def setup_dirs():
    os.makedirs("plots", exist_ok=True)

def visualize_comparisons(num_samples=5):
    raw_test_path = "data/raw/test"
    tiled_test_path = "data/tiled/test" 

    raw_files = sorted(glob.glob(os.path.join(raw_test_path, "*.png")))
    tiled_files = sorted(glob.glob(os.path.join(tiled_test_path, "*.png")))

    found = 0
    plt.figure(figsize=(18, 8))

    print(f"Searching for {num_samples} cases where Tiled > Raw...")

    for r_path, t_path in zip(raw_files, tiled_files):
        r_img = Image.open(r_path).convert('L')
        t_img = Image.open(t_path).convert('L')

        # Normalization
        r_tensor = torch.from_numpy(np.array(r_img)/255.0 - 0.5).unsqueeze(0).unsqueeze(0).float()
        t_tensor = torch.from_numpy(np.array(t_img)/255.0 - 0.5).unsqueeze(0).unsqueeze(0).float()

        label = int(r_path.split('_label_')[-1].split('.')[0])

        pred_raw = get_predictions("raw", r_tensor).item()
        pred_tiled = get_predictions("tiled", t_tensor).item()

        if pred_tiled == label and pred_raw != label:
            found += 1 
            
            plt.subplot(2, num_samples, found)
            plt.imshow(np.array(r_img), cmap='gray')
            plt.title(f"RAW (Wrong): {pred_raw}")
            plt.axis('off')

            plt.subplot(2, num_samples, found + num_samples)
            plt.imshow(np.array(t_img), cmap='gray')
            plt.title(f"TILED (Correct): {pred_tiled}\nActual: {label}")
            plt.axis('off')

            if found >= num_samples:
                break

    if found < num_samples:
        print(f"Only found {found} matches out of the requested {num_samples}.")
    
    if found > 0:
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/comparison_5_samples.png")
        plt.show()
    else:
        plt.close()                      

def plot_learning_curves():
    plt.figure(figsize=(10, 6))

    modes = [("raw", "blue"), ("tiled", "red")] 
    
    plot_found = False
    for mode, color in modes:
        path = f"results/history_{mode}.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                history = json.load(f)
            plt.plot(range(1, len(history) + 1), history, label=f'{mode.upper()} Loss', color=color, marker='o')
            plot_found = True
        else:
            print(f"Warning: Could not find history file at {path}")

    if plot_found:
        plt.title("Training Convergence: Raw vs Tiled")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plots/learning_curves.png")
        plt.show()

if __name__ == "__main__":
    setup_dirs()
    print("---Final Results---")
    evaluate_model("raw")
    evaluate_model("tiled") 
    visualize_comparisons(num_samples=5)
    plot_learning_curves()
