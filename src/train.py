#!/usr/bin/env python3
"""
Duality Offroad Semantic Segmentation Training Script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix, jaccard_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from datetime import datetime

# Class definitions for the segmentation task
CLASS_MAPPING = {
    0: 0,      # Background/Other
    100: 1,    # Trees
    200: 2,    # Lush Bushes
    300: 3,    # Dry Grass
    500: 4,    # Dry Bushes
    550: 5,    # Ground Clutter
    600: 6,    # Flowers
    700: 7,    # Logs
    800: 8,    # Rocks
    7100: 9,   # Landscape
    10000: 10  # Sky
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

NUM_CLASSES = len(CLASS_NAMES)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png'))
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert mask values to class indices
        mask_converted = np.zeros_like(mask)
        for original_val, class_idx in CLASS_MAPPING.items():
            mask_converted[mask == original_val] = class_idx
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask_converted)
            image = augmented['image']
            mask_converted = augmented['mask']
        
        return image, mask_converted.long()

def get_transforms():
    train_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def calculate_iou(outputs, masks, num_classes):
    outputs = torch.argmax(outputs, dim=1)
    iou_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (outputs == class_idx)
        true_class = (masks == class_idx)
        
        intersection = torch.sum(pred_class & true_class).float()
        union = torch.sum(pred_class | true_class).float()
        
        if union == 0:
            iou_scores.append(1.0 if torch.sum(true_class) == 0 else 0.0)
        else:
            iou_scores.append((intersection / union).item())
    
    return np.mean(iou_scores), iou_scores

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    train_img_dir = "data/train"
    train_mask_dir = "data/train"
    val_img_dir = "data/val"
    val_mask_dir = "data/val"
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, train_transform)
    val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_iou, _ = calculate_iou(outputs, masks, NUM_CLASSES)
            train_iou += batch_iou
            
            train_pbar.set_postfix({'loss': loss.item(), 'iou': batch_iou})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                batch_iou, _ = calculate_iou(outputs, masks, NUM_CLASSES)
                val_iou += batch_iou
                
                val_pbar.set_postfix({'loss': loss.item(), 'iou': batch_iou})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ious.append(avg_train_iou)
        val_ious.append(avg_val_iou)
        
        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou,
                'class_mapping': CLASS_MAPPING,
                'class_names': CLASS_NAMES,
            }, 'runs/best_model.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
        print(f"  Best Val IoU: {best_iou:.4f}")
        print("-" * 50)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'best_iou': best_iou,
        'num_epochs': num_epochs,
        'class_names': CLASS_NAMES,
    }
    
    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Val IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Val IoU')
    plt.title('IoU Progress')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('logs/training_curves.png')
    plt.close()
    
    print(f"Training completed! Best validation IoU: {best_iou:.4f}")
    print(f"Best model saved to: runs/best_model.pth")
    print(f"Training curves saved to: logs/training_curves.png")

if __name__ == "__main__":
    train_model()
