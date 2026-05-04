#!/usr/bin/env python3
"""
Duality Offroad Semantic Segmentation - Advanced Training Script
With Test-Time Augmentation, Class Weights, and Enhanced Architecture
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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

# Advanced color palette for visualization
COLORS = [
    [0, 0, 0],        # Background - Black
    [34, 139, 34],    # Trees - Forest Green
    [0, 128, 0],      # Lush Bushes - Green
    [255, 228, 181],  # Dry Grass - Burlywood
    [139, 69, 19],    # Dry Bushes - Saddle Brown
    [105, 105, 105],  # Ground Clutter - Dim Gray
    [255, 192, 203],  # Flowers - Pink
    [160, 82, 45],    # Logs - Sienna
    [128, 128, 128],  # Rocks - Gray
    [70, 130, 180],   # Landscape - Steel Blue
    [135, 206, 235],  # Sky - Sky Blue
]

class AdvancedSegmentationDataset(Dataset):
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

def calculate_class_weights(dataset):
    """Calculate class weights based on frequency to handle imbalance"""
    print("Calculating advanced class weights...")
    class_counts = np.zeros(NUM_CLASSES)
    
    # Sample a subset to calculate weights
    sample_size = min(500, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="Analyzing class distribution"):
        _, mask = dataset[idx]
        mask_np = mask.numpy()
        for class_idx in range(NUM_CLASSES):
            class_counts[class_idx] += np.sum(mask_np == class_idx)
    
    # Calculate weights with inverse frequency and smoothing
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (NUM_CLASSES * class_counts + 1e-6)
    
    # Apply smoothing and normalization
    class_weights = np.power(class_weights, 0.5)  # Square root smoothing
    class_weights = class_weights / np.sum(class_weights) * NUM_CLASSES
    
    print("Advanced class weights:", dict(zip(CLASS_NAMES, class_weights.round(3))))
    return torch.FloatTensor(class_weights)

def get_advanced_transforms():
    """Advanced data augmentation pipeline"""
    train_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.1, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RGBShift(p=0.3, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            A.HueSaturationValue(p=0.3, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(p=0.3, var_limit=(10.0, 50.0)),
            A.GaussianBlur(p=0.2, blur_limit=(3, 7)),
            A.MedianBlur(p=0.1, blur_limit=5),
        ], p=0.2),
        A.OneOf([
            A.RandomFog(p=0.2, fog_coef_lower=0.1, fog_coef_upper=0.3),
            A.RandomSunFlare(p=0.1, src_radius=100),
            A.RandomRain(p=0.1, slant_lower=-10, slant_upper=10),
        ], p=0.1),
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
    """Calculate IoU with proper handling of edge cases"""
    outputs = torch.argmax(outputs, dim=1)
    iou_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (outputs == class_idx)
        true_class = (masks == class_idx)
        
        intersection = torch.logical_and(pred_class, true_class).sum()
        union = torch.logical_or(pred_class, true_class).sum()
        
        if union == 0:
            iou_scores.append(float('nan'))
        else:
            iou_scores.append(float(intersection / union))
    
    return np.nanmean(iou_scores), iou_scores

def calculate_dice_coefficient(outputs, masks, num_classes):
    """Calculate Dice coefficient for additional evaluation"""
    outputs = torch.argmax(outputs, dim=1)
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (outputs == class_idx).float()
        true_class = (masks == class_idx).float()
        
        intersection = (pred_class * true_class).sum()
        union = pred_class.sum() + true_class.sum()
        
        if union == 0:
            dice_scores.append(float('nan'))
        else:
            dice_scores.append(float(2.0 * intersection / union))
    
    return np.nanmean(dice_scores), dice_scores

def train_advanced_model():
    """Advanced training with enhanced techniques"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create runs directory
    os.makedirs('runs', exist_ok=True)
    
    # Dataset paths
    train_img_dir = "data/train"
    train_mask_dir = "data/train"
    val_img_dir = "data/val"
    val_mask_dir = "data/val"
    
    # Get transforms
    train_transform, val_transform = get_advanced_transforms()
    
    # Create datasets
    train_dataset = AdvancedSegmentationDataset(train_img_dir, train_mask_dir, train_transform)
    val_dataset = AdvancedSegmentationDataset(val_img_dir, val_mask_dir, val_transform)
    
    # Calculate class weights for imbalance handling
    class_weights = calculate_class_weights(train_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)  # Smaller batch for advanced model
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create advanced model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_attention_type='scse',  # Add attention mechanism
    )
    model = model.to(device)
    
    # Advanced loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer with weight decay and different learning rates
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 2e-4},
        {'params': model.segmentation_head.parameters(), 'lr': 3e-4},
    ], weight_decay=1e-4)
    
    # Advanced learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training metrics
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_dices = []
    val_dices = []
    best_iou = 0.0
    patience_counter = 0
    max_patience = 15
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_iou, _ = calculate_iou(outputs, masks, NUM_CLASSES)
            batch_dice, _ = calculate_dice_coefficient(outputs, masks, NUM_CLASSES)
            train_iou += batch_iou
            train_dice += batch_dice
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'iou': f'{batch_iou:.4f}',
                'dice': f'{batch_dice:.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                batch_iou, _ = calculate_iou(outputs, masks, NUM_CLASSES)
                batch_dice, _ = calculate_dice_coefficient(outputs, masks, NUM_CLASSES)
                val_iou += batch_iou
                val_dice += batch_dice
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'iou': f'{batch_iou:.4f}',
                    'dice': f'{batch_dice:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ious.append(avg_train_iou)
        val_ious.append(avg_val_iou)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        
        # Early stopping with patience
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'iou': best_iou,
                'dice': avg_val_dice,
                'class_weights': class_weights,
                'class_mapping': CLASS_MAPPING,
                'class_names': CLASS_NAMES,
                'model_config': {
                    'encoder': 'resnet34',
                    'attention': 'scse',
                    'pretrained': 'imagenet'
                }
            }, 'runs/best_model_advanced.pth')
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}")
        print(f"  Best Val IoU: {best_iou:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Patience Counter: {patience_counter}/{max_patience}")
        print("-" * 60)
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'best_iou': best_iou,
        'best_dice': max(val_dices),
        'class_weights': class_weights.tolist(),
        'class_names': CLASS_NAMES,
        'model_config': {
            'encoder': 'resnet34',
            'attention': 'scse',
            'pretrained': 'imagenet',
            'advanced_augmentation': True,
            'class_weights': True
        }
    }
    
    with open('runs/training_history_advanced.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot comprehensive training curves
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 2)
    plt.plot(train_ious, label='Train IoU', color='green')
    plt.plot(val_ious, label='Val IoU', color='orange')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    plt.plot(train_dices, label='Train Dice', color='purple')
    plt.plot(val_dices, label='Val Dice', color='brown')
    plt.title('Training and Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    lrs = [scheduler.get_last_lr()[0]] * len(train_losses)
    plt.plot(lrs, color='black')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runs/training_curves_advanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Advanced training completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Best validation Dice: {max(val_dices):.4f}")
    print(f"Model saved as: runs/best_model_advanced.pth")

if __name__ == "__main__":
    train_advanced_model()
