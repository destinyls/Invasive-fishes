import os
import argparse
import json
import time
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from model import FishClassifier
from dataset import FishDataset

def evaluate_model(model, data_dir, output_dir='results', batch_size=32, image_size=224, num_workers=4):
    """Evaluate model on validation dataset with enhanced metrics"""
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Get all class names from directories
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    results = {
        'correct': 0,
        'total': 0,
        'class_accuracy': {},
        'predictions': {},
        'confidence_bins': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'confusion_matrix': None,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    # Prepare for metrics
    all_true_labels = []
    all_pred_labels = []
    all_similarities = []
    
    # Process each class
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Processing classes")):
        class_dir = os.path.join(data_dir, class_name)
        class_correct = 0
        class_total = 0
        results['class_accuracy'][class_name] = 0.0
        
        # Get all images for this class
        image_files = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"Class {class_name}", leave=False):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Get prediction
                prediction = model.predict(img_path)
                pred_class = prediction['predicted_class']
                similarity = prediction['similarity']
                
                # Add to confidence bins (0-0.1, 0.1-0.2, ..., 0.9-1.0)
                bin_idx = min(int(similarity * 10), 9)  # Clamp to 0-9
                confidence_bin = f"{bin_idx * 0.1:.1f}-{(bin_idx + 1) * 0.1:.1f}"
                results['confidence_bins'][confidence_bin]['total'] += 1
                if pred_class == class_name:
                    results['confidence_bins'][confidence_bin]['correct'] += 1
                
                # Log prediction details
                results['predictions'][img_path] = {
                    'true_class': class_name,
                    'predicted_class': pred_class,
                    'similarity': similarity,
                    'correct': pred_class == class_name,
                    'top_similarities': dict(sorted(prediction['all_similarities'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5])
                }
                
                # Update metrics
                is_correct = pred_class == class_name
                class_total += 1
                if is_correct:
                    class_correct += 1
                
                all_true_labels.append(class_name)
                all_pred_labels.append(pred_class)
                all_similarities.append(similarity)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Calculate class accuracy
        if class_total > 0:
            class_accuracy = class_correct / class_total
            results['class_accuracy'][class_name] = class_accuracy
            print(f"Class {class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")
        
        results['correct'] += class_correct
        results['total'] += class_total
    
    # Calculate overall metrics
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted'
        )
        
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        
        # Confusion matrix
        labels = sorted(class_names)
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
        results['confusion_matrix'] = cm.tolist()
        
        # Calculate calibration metrics (Confidence vs Accuracy)
        for bin_name, bin_data in results['confidence_bins'].items():
            if bin_data['total'] > 0:
                bin_data['accuracy'] = bin_data['correct'] / bin_data['total']
            else:
                bin_data['accuracy'] = 0
        
        # Create visualizations
        create_visualizations(results, class_names, all_similarities, output_dir)
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    results['execution_time'] = elapsed_time
    
    # Print summary
    print("\n===== Evaluation Results =====")
    print(f"Total images: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Overall accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_visualizations(results, class_names, similarities, output_dir):
    """Create enhanced visualizations for evaluation results"""
    # 1. Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = np.array(results['confusion_matrix'])
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot with seaborn for better visualization
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 2. Class accuracy bar plot with error bars
    plt.figure(figsize=(14, 6))
    class_acc = results['class_accuracy']
    sorted_classes = sorted(class_acc.keys(), key=lambda x: class_acc[x], reverse=True)
    accuracies = [class_acc[c] for c in sorted_classes]
    
    # Calculate error margin (95% confidence interval)
    class_totals = {cls: sum(1 for item in results['predictions'].values() if item['true_class'] == cls)
                   for cls in sorted_classes}
    error_margins = [1.96 * np.sqrt((acc * (1 - acc)) / class_totals[cls]) 
                    for acc, cls in zip(accuracies, sorted_classes)]
    
    plt.bar(range(len(sorted_classes)), accuracies, yerr=error_margins)
    plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy (with 95% Confidence Intervals)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'))
    
    # 3. Similarity distribution plot
    plt.figure(figsize=(10, 6))
    correct_predictions = [results['predictions'][p]['similarity'] for p in results['predictions'] 
                         if results['predictions'][p]['correct']]
    incorrect_predictions = [results['predictions'][p]['similarity'] for p in results['predictions'] 
                           if not results['predictions'][p]['correct']]
    
    plt.hist([correct_predictions, incorrect_predictions], bins=20, alpha=0.7, 
            label=['Correct', 'Incorrect'])
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    
    # 4. Reliability diagram (Calibration curve)
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    # Sort bins by confidence
    sorted_bins = sorted(results['confidence_bins'].items(), key=lambda x: float(x[0].split('-')[0]))
    
    for bin_name, bin_data in sorted_bins:
        if bin_data['total'] > 0:
            bin_accuracies.append(bin_data['accuracy'])
            # Use the midpoint of the bin as the confidence
            bin_mid = (float(bin_name.split('-')[0]) + float(bin_name.split('-')[1])) / 2
            bin_confidences.append(bin_mid)
            bin_sizes.append(bin_data['total'])
    
    plt.figure(figsize=(10, 6))
    
    # Plot the identity line (perfectly calibrated)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Plot calibration curve
    plt.scatter(bin_confidences, bin_accuracies, s=[x/5 for x in bin_sizes], alpha=0.8)
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model calibration')
    
    # Add bin counts as text
    for i, (x, y, size) in enumerate(zip(bin_confidences, bin_accuracies, bin_sizes)):
        plt.text(x, y, f"{size}", fontsize=9, ha='center', va='bottom')
    
    plt.xlabel('Confidence (Predicted Probability)')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    
    # 5. Most confused pairs
    class_confusion = {}
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                class_confusion[(true_class, pred_class)] = cm[i, j]
    
    # Get top confused pairs
    top_confused = sorted(class_confusion.items(), key=lambda x: x[1], reverse=True)[:10]
    
    plt.figure(figsize=(10, 6))
    pair_labels = [f"{true} → {pred}" for (true, pred), _ in top_confused]
    confusion_counts = [count for _, count in top_confused]
    
    plt.barh(range(len(pair_labels)), confusion_counts, color='salmon')
    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel('Count')
    plt.ylabel('True → Predicted')
    plt.title('Top 10 Most Confused Class Pairs')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'most_confused_pairs.png'))

def main():
    parser = argparse.ArgumentParser(description='Evaluate Fish Classification Model')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data directory with class subdirectories')
    parser.add_argument('--model_path', type=str, default='models/fish_classifier.pth',
                        help='Path to trained model')
    parser.add_argument('--template_dir', type=str, default='data/template',
                        help='Path to directory containing template images')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (when applicable)')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                      choices=['resnet50', 'resnet101', 'efficientnet_b2'],
                      help='Backbone network architecture')
    
    args = parser.parse_args()
    
    # Check if validation directory exists
    if not os.path.exists(args.val_dir):
        print(f"Validation directory not found: {args.val_dir}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    # Check if template directory exists
    if not os.path.exists(args.template_dir):
        print(f"Template directory not found: {args.template_dir}")
        return
    
    # Initialize model
    print(f"Loading model from {args.model_path}")
    model = FishClassifier(
        model_path=args.model_path,
        template_dir=args.template_dir,
        image_size=args.image_size
    )
    
    # Update backbone if specified
    if args.backbone != 'resnet50':
        print(f"Note: Backbone was set to {args.backbone}, but loaded model parameters will override this")
    
    # Run evaluation
    evaluate_model(
        model=model,
        data_dir=args.val_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 