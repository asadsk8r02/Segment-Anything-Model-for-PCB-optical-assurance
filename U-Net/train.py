import os
import cv2
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import shutil
import math

from unet_model import UNet, DISLoss 

# Dataset class to handle image and mask loading
class PCBImageDataset(Dataset):
    def __init__(self, img_folder, mask_folder, imsize=256):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.img_files = sorted(os.listdir(img_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.imsize = imsize

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
        
        img = cv2.imread(img_path) / 255.0
        img = cv2.resize(img, (self.imsize, self.imsize))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # Convert to (C, H, W) for PyTorch

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.resize(mask, (self.imsize, self.imsize))
        mask = np.expand_dims(mask, axis=0).astype(np.float32)  # Expand to (1, H, W)

        return torch.tensor(img), torch.tensor(mask)

# Create train, validation, and test splits
class DatasetSplitter:
    def __init__(self, images_path, masks_path, output_path, test_size=0.2, val_size=0.2):
        self.images_path = images_path
        self.masks_path = masks_path
        self.output_path = output_path
        self.test_size = test_size
        self.val_size = val_size

    def create_splits(self):
        sets = ['train', 'test', 'val']
        for set_name in sets:
            os.makedirs(os.path.join(self.output_path, set_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, set_name, 'masks'), exist_ok=True)

        images = sorted([img for img in os.listdir(self.images_path) if img.endswith('_image.jpg')])
        masks = sorted([mask for mask in os.listdir(self.masks_path) if mask.endswith('_mask.png')])

        images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=self.test_size, random_state=42)
        images_train, images_val, masks_train, masks_val = train_test_split(images_train, masks_train, test_size=self.val_size, random_state=42)

        def copy_files(files, source, dest):
            for file in files:
                shutil.copy(os.path.join(source, file), os.path.join(dest, file))

        copy_files(images_train, self.images_path, os.path.join(self.output_path, 'train', 'images'))
        copy_files(masks_train, self.masks_path, os.path.join(self.output_path, 'train', 'masks'))
        copy_files(images_test, self.images_path, os.path.join(self.output_path, 'test', 'images'))
        copy_files(masks_test, self.masks_path, os.path.join(self.output_path, 'test', 'masks'))
        copy_files(images_val, self.images_path, os.path.join(self.output_path, 'val', 'images'))
        copy_files(masks_val, self.masks_path, os.path.join(self.output_path, 'val', 'masks'))

        print(f"Dataset split: {len(images_train)} train, {len(images_test)} test, {len(images_val)} val")

# Training class for managing the training process
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, val_loader, epochs=30):
        self.model.to(self.device)
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

        return history

    def _train_one_epoch(self, data_loader):
        self.model.train()
        running_loss = 0.0

        for imgs, masks in data_loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        return running_loss / len(data_loader.dataset)

    def _validate(self, data_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for imgs, masks in data_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item() * imgs.size(0)

        return running_loss / len(data_loader.dataset)

# Prediction and visualization class
class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_mask = self.model(image)
            pred_mask = (pred_mask > 0.4).float()
            return pred_mask.squeeze().cpu().numpy()

    def display(self, display_list):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        for i, display_item in enumerate(display_list):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(display_item, cmap='gray')
            plt.axis('off')
        plt.show()

# Usage example
def main():
    images_path = "/path/to/images"
    masks_path = "/path/to/masks"
    output_path = "/path/to/output"

    # Create splits
    splitter = DatasetSplitter(images_path, masks_path, output_path)
    splitter.create_splits()

    # Datasets and DataLoaders
    train_dataset = PCBImageDataset(os.path.join(output_path, 'train', 'images'), os.path.join(output_path, 'train', 'masks'))
    val_dataset = PCBImageDataset(os.path.join(output_path, 'val', 'images'), os.path.join(output_path, 'val', 'masks'))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model, criterion, optimizer
    model = UNet(in_channels=3, out_channels=1)
    criterion = DISLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    trainer = ModelTrainer(model, criterion, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')
    history = trainer.train(train_loader, val_loader, epochs=30)

    # Load best model for prediction
    model.load_state_dict(torch.load('best_model.pth'))
    predictor = Predictor(model, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Example prediction
    sample_image = cv2.imread("/path/to/sample_image.jpg") / 255.0
    sample_image = cv2.resize(sample_image, (256, 256))
    sample_image = np.transpose(sample_image, (2, 0, 1))

    pred_mask = predictor.predict(sample_image)
    true_mask = cv2.imread("/path/to/sample_mask.png", cv2.IMREAD_GRAYSCALE) / 255.0
    true_mask = cv2.resize(true_mask, (256, 256))

    # Display the result
    predictor.display([sample_image.transpose(1, 2, 0), true_mask, pred_mask])

if __name__ == "__main__":
    main()
