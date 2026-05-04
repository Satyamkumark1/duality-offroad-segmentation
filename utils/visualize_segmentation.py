#!/usr/bin/env python3
"""
Visualization script for Duality Offroad Semantic Segmentation
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Class definitions
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

# High contrast colors for visualization
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

def convert_mask_to_colored(mask):
    """Convert grayscale mask to colored segmentation"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for original_val, class_idx in CLASS_MAPPING.items():
        colored_mask[mask == original_val] = COLORS[class_idx]
    
    return colored_mask

def create_legend():
    """Create a legend for the segmentation classes"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        color_normalized = [c/255.0 for c in color]
        ax.add_patch(plt.Rectangle((0, i), 1, 1, facecolor=color_normalized, edgecolor='black'))
        ax.text(1.1, i + 0.5, f'{i}: {class_name}', va='center', fontsize=10)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(CLASS_NAMES))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Segmentation Class Legend', fontsize=14, fontweight='bold')
    
    return fig

def visualize_single_image(image_path, mask_path=None, save_path=None):
    """Visualize a single image with optional segmentation mask"""
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if mask_path and os.path.exists(mask_path):
        # Load and convert mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        colored_mask = convert_mask_to_colored(mask)
        
        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask', fontsize=12)
        axes[1].axis('off')
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=12)
        axes[2].axis('off')
        
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes.imshow(image)
        axes.set_title('Original Image', fontsize=12)
        axes.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_dataset_stats(data_dir):
    """Visualize dataset statistics"""
    
    # Count images and masks
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    
    print(f"Dataset Statistics for {data_dir}:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total masks: {len(mask_files)}")
    
    # Analyze class distribution in masks
    class_counts = {class_name: 0 for class_name in CLASS_NAMES}
    total_pixels = 0
    
    for mask_file in tqdm(mask_files[:50]):  # Sample first 50 masks for speed
        mask_path = os.path.join(data_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        unique, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique, counts):
            if val in CLASS_MAPPING:
                class_idx = CLASS_MAPPING[val]
                class_counts[CLASS_NAMES[class_idx]] += count
                total_pixels += count
    
    # Convert to percentages
    class_percentages = {name: (count / total_pixels) * 100 for name, count in class_counts.items()}
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Class distribution pie chart
    labels = list(class_percentages.keys())
    sizes = list(class_percentages.values())
    colors_plot = [[c/255.0 for c in COLORS[i]] for i in range(len(labels))]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_plot, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    
    # Class distribution bar chart
    ax2.bar(labels, sizes, color=colors_plot)
    ax2.set_title('Class Pixel Counts', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage of Total Pixels')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'logs/{os.path.basename(data_dir)}_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset statistics saved to: logs/{os.path.basename(data_dir)}_stats.png")
    
    return class_percentages

def create_training_curves_plot():
    """Create training curves visualization if training history exists"""
    
    history_path = 'logs/training_history.json'
    if not os.path.exists(history_path):
        print("No training history found. Run training first.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_losses'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU curves
    axes[0, 1].plot(history['train_ious'], label='Train IoU', color='green')
    axes[0, 1].plot(history['val_ious'], label='Val IoU', color='orange')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Loss difference
    loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
    axes[1, 0].plot(loss_diff, color='purple')
    axes[1, 0].set_title('Val Loss - Train Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # IoU difference
    iou_diff = np.array(history['val_ious']) - np.array(history['train_ious'])
    axes[1, 1].plot(iou_diff, color='brown')
    axes[1, 1].set_title('Val IoU - Train IoU')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU Difference')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('logs/detailed_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Detailed training curves saved to: logs/detailed_training_curves.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('--image', type=str, help='Path to single image to visualize')
    parser.add_argument('--mask', type=str, help='Path to corresponding mask')
    parser.add_argument('--save', type=str, help='Path to save visualization')
    parser.add_argument('--stats', type=str, help='Path to dataset directory for statistics')
    parser.add_argument('--legend', action='store_true', help='Create class legend')
    parser.add_argument('--curves', action='store_true', help='Create training curves plot')
    
    args = parser.parse_args()
    
    if args.legend:
        fig = create_legend()
        plt.savefig('logs/class_legend.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Class legend saved to: logs/class_legend.png")
    
    if args.curves:
        create_training_curves_plot()
    
    if args.image:
        visualize_single_image(args.image, args.mask, args.save)
    
    if args.stats:
        visualize_dataset_stats(args.stats)

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    main()
