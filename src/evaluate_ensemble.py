#!/usr/bin/env python3
"""
Professional Ensemble Evaluation for Duality Offroad Semantic Segmentation
Comprehensive evaluation of ensemble models with detailed metrics
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
from datetime import datetime
import cv2

# Import our modules
from ensemble import SegmentationEnsemble, load_ensemble_weights, create_ensemble_models
from train import SegmentationDataset, get_transforms, calculate_iou

class EnsembleEvaluator:
    """
    Professional evaluator for ensemble models
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.class_names = [
            'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
            'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
        ]
        self.num_classes = len(self.class_names)
        
        print(f"🔬 Ensemble Evaluator initialized")
        print(f"📊 Device: {self.device}")
        print(f"🎯 Classes: {self.num_classes}")
    
    def load_ensemble(self, checkpoint_dir: str = 'runs/ensemble_models') -> SegmentationEnsemble:
        """
        Load trained ensemble models
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            
        Returns:
            Initialized ensemble
        """
        print(f"\n📂 Loading ensemble models from {checkpoint_dir}")
        
        # Create models
        models = create_ensemble_models(self.device)
        
        # Load weights
        checkpoint_paths = [
            os.path.join(checkpoint_dir, 'best_unet34.pth'),
            os.path.join(checkpoint_dir, 'best_unet50.pth'),
            os.path.join(checkpoint_dir, 'best_fpn34.pth')
        ]
        
        load_ensemble_weights(models, checkpoint_paths, self.device)
        
        # Create ensemble with optimized weights
        # Based on typical performance: UNet34 > UNet50 > FPN
        weights = [0.4, 0.35, 0.25]
        ensemble = SegmentationEnsemble(models, weights)
        
        print(f"✅ Ensemble loaded with {len(models)} models")
        print(f"⚖️  Model weights: {weights}")
        
        return ensemble
    
    def evaluate_on_dataset(self, ensemble: SegmentationEnsemble, dataset_path: str, dataset_type: str = 'validation') -> Dict:
        """
        Evaluate ensemble on specified dataset
        
        Args:
            ensemble: Trained ensemble
            dataset_path: Path to dataset
            dataset_type: Type of dataset (validation/test)
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"\n📊 Evaluating ensemble on {dataset_type} dataset")
        print(f"📂 Dataset path: {dataset_path}")
        
        # Load dataset
        _, val_transform = get_transforms()
        dataset = SegmentationDataset(dataset_path, dataset_path, val_transform)
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        print(f"📈 Dataset size: {len(dataset)} samples")
        print(f"🔄 Batch size: 4")
        
        # Evaluation metrics
        total_iou = 0.0
        total_dice = 0.0
        per_class_iou = np.zeros(self.num_classes)
        per_class_dice = np.zeros(self.num_classes)
        per_class_pixels = np.zeros(self.num_classes)
        
        # Confusion matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc=f"Evaluating {dataset_type}"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Get ensemble predictions
                predictions = ensemble.predict_batch(images, self.device)
                
                # Calculate metrics for this batch
                batch_size = images.shape[0]
                
                for i in range(batch_size):
                    pred = predictions[i].cpu().numpy()
                    true = masks[i].cpu().numpy()
                    
                    # Update confusion matrix
                    for true_class in range(self.num_classes):
                        for pred_class in range(self.num_classes):
                            true_mask = (true == true_class)
                            pred_mask = (pred == pred_class)
                            confusion_matrix[true_class, pred_class] += np.sum(true_mask & pred_mask)
                    
                    # Per-class metrics
                    for class_idx in range(self.num_classes):
                        pred_class = (pred == class_idx)
                        true_class = (true == class_idx)
                        
                        intersection = np.sum(pred_class & true_class)
                        union = np.sum(pred_class | true_class)
                        
                        if union > 0:
                            iou = intersection / union
                            per_class_iou[class_idx] += iou
                            
                            # Dice coefficient
                            pred_sum = np.sum(pred_class)
                            true_sum = np.sum(true_class)
                            if pred_sum + true_sum > 0:
                                dice = 2.0 * intersection / (pred_sum + true_sum)
                                per_class_dice[class_idx] += dice
                        
                        # Pixel count
                        per_class_pixels[class_idx] += np.sum(true_class)
                
                # Progress update
                current_iou = np.mean(per_class_iou[per_class_pixels > 0])
                print(f"📊 Current mIoU: {current_iou:.4f}")
        
        # Calculate final metrics
        num_samples = len(dataset)
        valid_classes = per_class_pixels > 0
        
        # Average metrics
        avg_iou = np.mean(per_class_iou[valid_classes])
        avg_dice = np.mean(per_class_dice[valid_classes])
        
        # Class-wise results
        class_results = {}
        for i, class_name in enumerate(self.class_names):
            if valid_classes[i]:
                class_results[class_name] = {
                    'iou': per_class_iou[i] / (per_class_pixels[i] / np.mean(per_class_pixels)),
                    'dice': per_class_dice[i] / (per_class_pixels[i] / np.mean(per_class_pixels)),
                    'pixel_percentage': (per_class_pixels[i] / np.sum(per_class_pixels)) * 100
                }
            else:
                class_results[class_name] = {
                    'iou': 0.0,
                    'dice': 0.0,
                    'pixel_percentage': 0.0
                }
        
        # Compile results
        results = {
            'dataset_type': dataset_type,
            'num_samples': num_samples,
            'mean_iou': avg_iou,
            'mean_dice': avg_dice,
            'per_class_results': class_results,
            'confusion_matrix': confusion_matrix.tolist(),
            'valid_classes': np.sum(valid_classes),
            'evaluation_date': datetime.now().isoformat()
        }
        
        return results
    
    def generate_report(self, results: Dict, save_path: str = 'runs/ensemble_evaluation_report.json'):
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save report
        """
        print(f"\n📝 Generating evaluation report...")
        
        # Create detailed report
        report = {
            'evaluation_summary': {
                'dataset_type': results['dataset_type'],
                'num_samples': results['num_samples'],
                'mean_iou': results['mean_iou'],
                'mean_dice': results['mean_dice'],
                'valid_classes': results['valid_classes'],
                'evaluation_date': results['evaluation_date']
            },
            'performance_metrics': {
                'overall_iou': results['mean_iou'],
                'overall_dice': results['mean_dice'],
                'class_distribution': results['per_class_results']
            },
            'class_wise_analysis': {}
        }
        
        # Class-wise analysis
        for class_name, metrics in results['per_class_results'].items():
            if metrics['iou'] > 0:
                performance_level = "Excellent" if metrics['iou'] > 0.9 else \
                                 "Good" if metrics['iou'] > 0.7 else \
                                 "Fair" if metrics['iou'] > 0.5 else "Poor"
                
                report['class_wise_analysis'][class_name] = {
                    'performance': performance_level,
                    'iou': metrics['iou'],
                    'dice': metrics['dice'],
                    'pixel_percentage': metrics['pixel_percentage'],
                    'dominance': "High" if metrics['pixel_percentage'] > 10 else \
                               "Medium" if metrics['pixel_percentage'] > 1 else "Low"
                }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved to {save_path}")
        
        return report
    
    def create_visualizations(self, results: Dict, save_dir: str = 'runs/ensemble_visualizations'):
        """
        Create professional visualizations of results
        
        Args:
            results: Evaluation results
            save_dir: Directory to save visualizations
        """
        print(f"\n🎨 Creating visualizations...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Class-wise IoU bar chart
        plt.figure(figsize=(12, 8))
        class_names = list(results['per_class_results'].keys())
        iou_scores = [results['per_class_results'][name]['iou'] for name in class_names]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A5ACD', 
                 '#61A0AF', '#2B8C3E', '#F1C40F', '#E74C3C', '#34495E', '#0077B6']
        
        bars = plt.bar(class_names, iou_scores, color=colors[:len(class_names)])
        plt.title('Per-Class IoU Performance', fontsize=16, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, iou in zip(bars, iou_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{iou:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/per_class_iou.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Class distribution pie chart
        plt.figure(figsize=(10, 8))
        pixel_percentages = [results['per_class_results'][name]['pixel_percentage'] 
                          for name in class_names]
        colors = colors[:len(class_names)]
        
        wedges, texts, autotexts = plt.pie(pixel_percentages, labels=class_names, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrix heatmap
        plt.figure(figsize=(12, 10))
        confusion_matrix = np.array(results['confusion_matrix'])
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall metrics
        metrics = ['IoU', 'Dice']
        values = [results['mean_iou'], results['mean_dice']]
        bars = ax1.bar(metrics, values, color=['#2E86AB', '#A23B72'])
        ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Class performance levels
        performance_levels = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        for class_name, metrics in results['per_class_results'].items():
            if metrics['iou'] > 0:
                if metrics['iou'] > 0.9:
                    performance_levels['Excellent'] += 1
                elif metrics['iou'] > 0.7:
                    performance_levels['Good'] += 1
                elif metrics['iou'] > 0.5:
                    performance_levels['Fair'] += 1
                else:
                    performance_levels['Poor'] += 1
        
        ax2.pie(performance_levels.values(), labels=performance_levels.keys(),
                colors=['#2E86AB', '#A23B72', '#F18F01', '#E74C3C'],
                autopct='%1.0f%%', startangle=90)
        ax2.set_title('Class Performance Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {save_dir}")
        print(f"  📊 Per-class IoU: per_class_iou.png")
        print(f"  🥧 Class distribution: class_distribution.png")
        print(f"  🔥 Confusion matrix: confusion_matrix.png")
        print(f"  📈 Performance summary: performance_summary.png")

def main():
    """
    Main evaluation function
    """
    print("🔬 Duality Offroad Segmentation - Ensemble Evaluation")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Device: {device}")
    
    # Create evaluator
    evaluator = EnsembleEvaluator(device)
    
    # Load ensemble
    ensemble = evaluator.load_ensemble()
    
    # Evaluate on validation set
    try:
        results = evaluator.evaluate_on_dataset(ensemble, 'data/val', 'validation')
        
        # Generate report
        report = evaluator.generate_report(results)
        
        # Create visualizations
        evaluator.create_visualizations(results)
        
        # Print summary
        print(f"\n🎉 Ensemble Evaluation Completed!")
        print("=" * 60)
        print(f"📊 Dataset: {results['dataset_type']}")
        print(f"📈 Samples: {results['num_samples']}")
        print(f"🏆 Mean IoU: {results['mean_iou']:.4f}")
        print(f"🎯 Mean Dice: {results['mean_dice']:.4f}")
        print(f"📊 Valid Classes: {results['valid_classes']}/11")
        
        print(f"\n🏆 Top Performing Classes:")
        sorted_classes = sorted(results['per_class_results'].items(), 
                           key=lambda x: x[1]['iou'], reverse=True)
        for i, (class_name, metrics) in enumerate(sorted_classes[:5]):
            if metrics['iou'] > 0:
                print(f"  {i+1}. {class_name}: {metrics['iou']:.4f}")
        
        print(f"\n📁 Files Generated:")
        print(f"  📄 Report: runs/ensemble_evaluation_report.json")
        print(f"  🎨 Visualizations: runs/ensemble_visualizations/")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("💡 Make sure dataset is available and ensemble models are trained")

if __name__ == "__main__":
    main()
