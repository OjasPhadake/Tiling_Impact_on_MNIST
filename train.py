import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import glob
import json
from models.mlp_baseline import MNIST_MLP

class LocalFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(os.path.join(folder_path, "*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('L')

        img_np = np.array(image, dtype = np.float32)
        img_np /= 255.0
        img_np = (img_np - 0.5) / 0.5

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        label = int(img_path.split('_label_')[-1].split('.')[0])
        return img_tensor, label

def train(mode="raw", epochs=10):
    data_path = f"data/{mode}/train"
    train_loader = DataLoader(LocalFolderDataset(data_path), batch_size=32, shuffle=True)
    
    model = MNIST_MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = []
    print(f"\nStarting Training: {mode.upper()}")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/mlp_{mode}.pth")
    with open(f"results/history_{mode}.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    train(mode="raw", epochs=10)
    print("-" * 30)
    train(mode="tiled", epochs=10)