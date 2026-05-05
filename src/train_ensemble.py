#!/usr/bin/env python3
"""
Professional Ensemble Training for Duality Offroad Semantic Segmentation
Trains multiple models with different architectures for superior performance
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from datetime import datetime

# Import our existing classes
from train import SegmentationDataset, CLASS_MAPPING, CLASS_NAMES, NUM_CLASSES, calculate_iou

class EnsembleTrainer:
    """
    Professional trainer for multiple model architectures
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.models_config = {
            'unet34': {
                'name': 'UNet + ResNet34',
                'model_class': smp.Unet,
                'params': {
                    'encoder_name': 'resnet34',
                    'encoder_weights': 'imagenet',
                    'in_channels': 3,
                    'classes': NUM_CLASSES
                },
                'weight': 0.4  # Highest weight - baseline model
            },
            'unet50': {
                'name': 'UNet + ResNet50',
                'model_class': smp.Unet,
                'params': {
                    'encoder_name': 'resnet50',
                    'encoder_weights': 'imagenet',
                    'in_channels': 3,
                    'classes': NUM_CLASSES
                },
                'weight': 0.35  # Second highest
            },
            'fpn34': {
                'name': 'FPN + ResNet34',
                'model_class': smp.FPN,
                'params': {
                    'encoder_name': 'resnet34',
                    'encoder_weights': 'imagenet',
                    'in_channels': 3,
                    'classes': NUM_CLASSES
                },
                'weight': 0.25  # Feature pyramid network
            }
        }
        
        print(f"🚀 Ensemble Trainer initialized")
        print(f"📊 Models to train: {len(self.models_config)}")
        for key, config in self.models_config.items():
            print(f"  - {config['name']} (weight: {config['weight']})")
    
    def get_transforms(self):
        """Get data transforms for ensemble training"""
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
    
    def create_model(self, config_key: str):
        """Create model from configuration"""
        config = self.models_config[config_key]
        model = config['model_class'](**config['params'])
        return model.to(self.device)
    
    def train_single_model(self, model_key: str, train_loader, val_loader, num_epochs: int = 20):
        """
        Train a single model in the ensemble
        
        Args:
            model_key: Key for model configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Trained model and training history
        """
        print(f"\n🎯 Training {self.models_config[model_key]['name']}...")
        
        # Create model
        model = self.create_model(model_key)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training history
        train_losses = []
        val_losses = []
        val_ious = []
        best_iou = 0.0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for images, masks in train_pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for images, masks in val_pbar:
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    batch_iou, _ = calculate_iou(outputs, masks, NUM_CLASSES)
                    val_iou += batch_iou
                    
                    val_pbar.set_postfix({'loss': loss.item(), 'iou': batch_iou})
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                os.makedirs('runs/ensemble_models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou,
                    'model_type': model_key,
                    'model_name': self.models_config[model_key]['name']
                }, f'runs/ensemble_models/best_{model_key}.pth')
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val IoU: {avg_val_iou:.4f}")
            print(f"  Best IoU: {best_iou:.4f}")
            print("-" * 50)
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_ious': val_ious,
            'best_iou': best_iou,
            'model_type': model_key,
            'model_name': self.models_config[model_key]['name']
        }
        
        with open(f'runs/ensemble_models/{model_key}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ {self.models_config[model_key]['name']} training completed!")
        print(f"🏆 Best IoU: {best_iou:.4f}")
        
        return model, history
    
    def train_ensemble(self, train_loader, val_loader, num_epochs: int = 20):
        """
        Train all models in the ensemble
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs per model
            
        Returns:
            Dictionary with all trained models and their histories
        """
        print(f"\n🚀 Starting Ensemble Training")
        print(f"📊 Total models: {len(self.models_config)}")
        print(f"🎯 Epochs per model: {num_epochs}")
        print("=" * 60)
        
        trained_models = {}
        training_histories = {}
        
        # Train each model
        for model_key in self.models_config.keys():
            model, history = self.train_single_model(model_key, train_loader, val_loader, num_epochs)
            trained_models[model_key] = model
            training_histories[model_key] = history
        
        # Create ensemble summary
        ensemble_summary = {
            'models_trained': list(trained_models.keys()),
            'model_weights': {key: config['weight'] for key, config in self.models_config.items()},
            'individual_ious': {key: history['best_iou'] for key, history in training_histories.items()},
            'training_date': datetime.now().isoformat(),
            'total_models': len(self.models_config)
        }
        
        # Save ensemble summary
        with open('runs/ensemble_models/ensemble_summary.json', 'w') as f:
            json.dump(ensemble_summary, f, indent=2)
        
        print(f"\n🎉 Ensemble Training Completed!")
        print(f"📊 Models trained: {list(trained_models.keys())}")
        print(f"🏆 Individual IoUs:")
        for model_key, iou in ensemble_summary['individual_ious'].items():
            model_name = self.models_config[model_key]['name']
            weight = self.models_config[model_key]['weight']
            print(f"  {model_name}: {iou:.4f} (weight: {weight})")
        
        return trained_models, training_histories

def main():
    """
    Main function to train ensemble models
    """
    print("🏆 Duality Offroad Segmentation - Ensemble Training")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('runs/ensemble_models', exist_ok=True)
    
    # Get data transforms
    trainer = EnsembleTrainer(device)
    train_transform, val_transform = trainer.get_transforms()
    
    # Create datasets
    try:
        train_dataset = SegmentationDataset("data/train", "data/train", train_transform)
        val_dataset = SegmentationDataset("data/val", "data/val", val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
        
        print(f"📊 Dataset loaded:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        # Train ensemble
        trained_models, histories = trainer.train_ensemble(train_loader, val_loader, num_epochs=15)
        
        print(f"\n✅ All ensemble models trained successfully!")
        print(f"📁 Models saved in: runs/ensemble_models/")
        print(f"📊 Ensemble ready for evaluation!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 Make sure dataset is properly organized in data/ directory")

if __name__ == "__main__":
    main()
