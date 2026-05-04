#!/usr/bin/env python3
"""
Duality Offroad Semantic Segmentation Testing Script
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
import seaborn as sns

# Class definitions (same as training)
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

# Colors for visualization (high contrast)
COLORS = [
    [0, 0, 0],        # Background - Black
    [34, 139, 34],    # Trees - Forest Green
    [0, 255, 0],      # Lush Bushes - Lime Green
    [255, 255, 0],    # Dry Grass - Yellow
    [255, 165, 0],    # Dry Bushes - Orange
    [139, 69, 19],    # Ground Clutter - Saddle Brown
    [255, 192, 203],  # Flowers - Pink
    [128, 128, 128],  # Logs - Gray
    [105, 105, 105],  # Rocks - Dim Gray
    [139, 90, 43],    # Landscape - Brown
    [135, 206, 235]   # Sky - Sky Blue
]

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image name
        image_name = self.images[idx]
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, image_name

def get_test_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def mask_to_colored_image(mask):
    """Convert class indices to colored image for visualization"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        colored_mask[mask == class_idx] = color
    return colored_mask

def calculate_class_iou(pred_mask, true_mask, num_classes):
    """Calculate IoU for each class"""
    ious = []
    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx)
        true_class = (true_mask == class_idx)
        
        intersection = np.sum(pred_class & true_class)
        union = np.sum(pred_class | true_class)
        
        if union == 0:
            ious.append(1.0 if np.sum(true_class) == 0 else 0.0)
        else:
            ious.append(intersection / union)
    
    return ious

def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Don't use pretrained weights
        in_channels=3,
        classes=NUM_CLASSES,
    )
    
    # Load checkpoint
    checkpoint_path = "runs/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run training first to generate the model.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Best validation IoU: {checkpoint['iou']:.4f}")
    
    # Test data path
    test_img_dir = "data/testImages"
    
    if not os.path.exists(test_img_dir):
        print(f"Error: Test images directory not found at {test_img_dir}")
        print("Please download and organize the dataset first.")
        return
    
    # Create test dataset and dataloader
    test_transform = get_test_transform()
    test_dataset = TestDataset(test_img_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"Found {len(test_dataset)} test images")
    
    # Create output directories
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    
    # Test loop
    all_predictions = []
    all_image_names = []
    class_ious = [[] for _ in range(NUM_CLASSES)]
    
    with torch.no_grad():
        for i, (images, image_names) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            image_name = image_names[0]
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert to numpy
            pred_mask = predictions.cpu().numpy()[0]
            
            # Save prediction
            all_predictions.append(pred_mask)
            all_image_names.append(image_name)
            
            # Save colored prediction
            colored_pred = mask_to_colored_image(pred_mask)
            pred_path = f"results/predictions/{image_name}"
            cv2.imwrite(pred_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
            
            # Create visualization with original image
            original_image = cv2.imread(os.path.join(test_img_dir, image_name))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Resize prediction to match original image
            pred_resized = cv2.resize(colored_pred, (original_image.shape[1], original_image.shape[0]))
            
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(pred_resized)
            axes[1].set_title('Predicted Segmentation')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"results/visualizations/{image_name.replace('.jpg', '.png').replace('.jpeg', '.png')}", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} images")
    
    print(f"Testing completed! Results saved to:")
    print(f"  - Predictions: results/predictions/")
    print(f"  - Visualizations: results/visualizations/")
    
    # Generate summary report
    generate_summary_report(all_image_names, checkpoint)

def generate_summary_report(image_names, checkpoint):
    """Generate a summary report of the testing results"""
    
    report = {
        'model_info': {
            'architecture': 'UNet with ResNet34 encoder',
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'best_validation_iou': checkpoint['iou'],
            'training_epoch': checkpoint['epoch']
        },
        'test_results': {
            'num_test_images': len(image_names),
            'test_images': image_names,
            'output_directories': {
                'predictions': 'results/predictions/',
                'visualizations': 'results/visualizations/',
                'metrics': 'results/metrics/'
            }
        },
        'class_mapping': CLASS_MAPPING
    }
    
    # Save report
    with open('results/test_summary.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Duality Offroad Segmentation - Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Duality Offroad Semantic Segmentation</h1>
            <h2>Test Results Report</h2>
        </div>
        
        <div class="section">
            <h3>Model Information</h3>
            <div class="metric"><strong>Architecture:</strong> UNet with ResNet34 encoder</div>
            <div class="metric"><strong>Number of Classes:</strong> {NUM_CLASSES}</div>
            <div class="metric"><strong>Best Validation IoU:</strong> {checkpoint['iou']:.4f}</div>
            <div class="metric"><strong>Training Epoch:</strong> {checkpoint['epoch']}</div>
        </div>
        
        <div class="section">
            <h3>Test Results</h3>
            <div class="metric"><strong>Number of Test Images:</strong> {len(image_names)}</div>
            <div class="metric"><strong>Predictions Saved:</strong> results/predictions/</div>
            <div class="metric"><strong>Visualizations Saved:</strong> results/visualizations/</div>
        </div>
        
        <div class="section">
            <h3>Class Definitions</h3>
            <div class="metric">
    """
    
    for i, class_name in enumerate(CLASS_NAMES):
        html_content += f"<strong>{i}:</strong> {class_name}<br>"
    
    html_content += f"""
            </div>
        </div>
        
        <div class="section">
            <h3>Sample Results</h3>
            <div class="image-grid">
    """
    
    # Add first 5 sample images
    for i, image_name in enumerate(image_names[:5]):
        viz_name = image_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        html_content += f"""
                <div class="image-item">
                    <h4>{image_name}</h4>
                    <img src="visualizations/{viz_name}" alt="{image_name}">
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open('results/test_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"Summary report saved to:")
    print(f"  - JSON: results/test_summary.json")
    print(f"  - HTML: results/test_report.html")

if __name__ == "__main__":
    test_model()
