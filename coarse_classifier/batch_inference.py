import os
import argparse
import json
import time
import statistics
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from collections import defaultdict

from model import FishClassifier

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class InferenceDataset(Dataset):
    """Dataset for batch inference with improved data handling"""
    def __init__(self, data_dir, transform=transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.true_labels = []
        self.class_counts = defaultdict(int)
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transform
        
        # Find all images and their corresponding class labels
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_name))
                        self.true_labels.append(class_name)
                        self.class_counts[class_name] += 1
        
        self.classes = sorted(list(self.class_counts.keys()))
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        
        # Print class distribution
        for class_name, count in sorted(self.class_counts.items()):
            print(f"  {class_name}: {count} images")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        try:
            # Load image without converting to RGB first to preserve alpha channel
            img = Image.open(img_path)
            
            # Process background removal for 4-channel images (maintain consistent with training)
            img = self._remove_background(img)
            
            if self.transform:
                img_tensor = self.transform(img)
            
            return {
                'image': img_tensor,
                'path': img_path,
                'true_class': class_name
            }
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return zeros as a placeholder
            return {
                'image': torch.zeros(3, 224, 224),
                'path': img_path,
                'true_class': class_name
            }
    
    def _remove_background(self, img):
        """
        Remove background from image using the alpha/mask channel.
        For 4-channel images, the 4th channel is treated as a mask.
        
        Args:
            img (PIL.Image): The input image with possible alpha/mask channel
            
        Returns:
            PIL.Image: RGB image with background removed
        """
        # Check if image has an alpha channel (4 channels)
        if img.mode == 'RGBA' or len(img.getbands()) == 4:
            # For webp with 4 channels, split the channels
            try:
                # Split the channels
                bands = img.split()
                if len(bands) == 4:
                    r, g, b, mask = bands
                    
                    # Apply mask to keep only foreground
                    rgb_masked = Image.new("RGB", img.size, (0, 0, 0))
                    rgb_masked.paste(img.convert('RGB'), mask=mask)
                    
                    return rgb_masked
                else:
                    return img.convert('RGB')
            except Exception as e:
                print(f"Error processing image with mask: {e}")
                return img.convert('RGB')
        else:
            # If not a 4-channel image, just convert to RGB
            return img.convert('RGB')

class BatchInferenceModel:
    """Model wrapper for efficient batch inference with the model"""
    def __init__(self, model_path, reference_dir, image_size=224, batch_size=32, num_workers=4, similarity_threshold=0.9):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = FishClassifier(
            model_path=model_path,
            image_size=image_size
        )
        
        # Setup parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.similarity_threshold = similarity_threshold
        
        # Extract reference features
        self.reference_features = self._extract_reference_features(reference_dir)
        self.classes = sorted(list(self.reference_features.keys()))

    def _extract_reference_features(self, reference_dir):
        """Extract features from reference images for each class"""
        print(f"Extracting reference features from {reference_dir}")
        reference_features = {}
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get all class directories
        for class_name in os.listdir(reference_dir):
            class_dir = os.path.join(reference_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Get all images in this class directory
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if not images:
                print(f"No images found for class {class_name}")
                continue
                
            # Extract features for all images
            class_features = []
            for img_file in images:
                image_path = os.path.join(class_dir, img_file)
                try:
                    # Load image without converting to RGB first to preserve alpha channel
                    img = Image.open(image_path)
                    
                    # Process background removal for 4-channel images (maintain consistent with training)
                    img = self._remove_background(img)
                    
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        # Extract features and pass through target backbone and projector
                        features = self.model.target_backbone(img_tensor)
                        features = self.model.target_projector(features)
                        features = F.normalize(features, dim=1, p=2)
                    
                    class_features.append(features)
                except Exception as e:
                    print(f"Error extracting features from {image_path}: {e}")
            
            if class_features:
                # Store all features for this class instead of averaging
                class_features = torch.cat(class_features, dim=0)
                reference_features[class_name] = class_features
                print(f"Extracted features for class {class_name} from {len(class_features)} images")
        
        print(f"Extracted reference features for {len(reference_features)} classes")
        return reference_features
    
    def _remove_background(self, img):
        """
        Remove background from image using the alpha/mask channel.
        For 4-channel images, the 4th channel is treated as a mask.
        
        Args:
            img (PIL.Image): The input image with possible alpha/mask channel
            
        Returns:
            PIL.Image: RGB image with background removed
        """
        # Check if image has an alpha channel (4 channels)
        if img.mode == 'RGBA' or len(img.getbands()) == 4:
            # For webp with 4 channels, split the channels
            try:
                # Split the channels
                bands = img.split()
                if len(bands) == 4:
                    r, g, b, mask = bands
                    
                    # Apply mask to keep only foreground
                    rgb_masked = Image.new("RGB", img.size, (0, 0, 0))
                    rgb_masked.paste(img.convert('RGB'), mask=mask)
                    
                    return rgb_masked
                else:
                    return img.convert('RGB')
            except Exception as e:
                print(f"Error processing image with mask: {e}")
                return img.convert('RGB')
        else:
            # If not a 4-channel image, just convert to RGB
            return img.convert('RGB')
        
    def process_batch(self, batch):
        """Process a batch of images and compare with reference features"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Extract images from batch
        images = batch['image'].to(self.device)
        paths = batch['path']
        true_classes = batch['true_class']
        
        results = []
        
        # Extract features
        with torch.no_grad():
            # Get features from online backbone
            features = self.model.online_backbone(images)
            
            # Compare with reference features
            for i in range(len(images)):
                img_feature = features[i:i+1]  # Keep batch dimension
                prediction = self._compare_with_references(img_feature)
                prediction['path'] = paths[i]
                prediction['true_class'] = true_classes[i]
                prediction['correct'] = prediction['predicted_class'] == true_classes[i]
                results.append(prediction)
        
        return results
    
    def _compare_with_references(self, feature):
        """Compare features with reference class features using standard BYOL approach"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Pass features through online projector and predictor
        with torch.no_grad():
            online_proj = self.model.online_projector(feature)
            online_pred = self.model.online_predictor(online_proj)
            # Normalize the online prediction
            online_pred = F.normalize(online_pred, dim=1, p=2)
        
        # Compare with reference features (which are already normalized)
        similarities, full_similarities = {}, {}
        for class_name, class_reference_features in self.reference_features.items():
            # Calculate BYOL loss with each reference feature in this class
            similarity_list = []
            for i in range(class_reference_features.size(0)):
                target_proj = class_reference_features[i:i+1]  # Already normalized
                
                # Calculate cosine similarity (dot product of normalized vectors)
                cos_sim = (online_pred * target_proj).sum(dim=1)
                
                similarity_list.append(cos_sim.item())
            
            # Convert BYOL loss to similarity (lower loss = higher similarity)
            # BYOL loss range is [0,2] where 0 is perfect match
            # Convert to similarity range [0,1] where 1 is perfect match
            full_similarities[class_name] = similarity_list
            similarities[class_name] = max(similarity_list)
         
        # Find the most similar class
        predicted_class, max_similarity = max(similarities.items(), key=lambda x: x[1])
        
        # Check if similarity is below threshold
        if max_similarity < self.similarity_threshold:
            predicted_class = "unknown"
        
        return {
            'predicted_class': predicted_class,
            'similarity': max_similarity,
            'all_similarities': similarities,
            'full_similarities': full_similarities  # Include full similarity distributions
        }
    
    def evaluate(self, data_dir, output_dir='results/batch'):
        """Run batch inference on a directory of images"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory for similarity distribution plots
        similarity_plots_dir = os.path.join(output_dir, 'similarity_distributions')
        os.makedirs(similarity_plots_dir, exist_ok=True)
        
        # Setup dataset and dataloader
        dataset = InferenceDataset(
            data_dir=data_dir,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        start_time = time.time()
        
        # Initialize results
        all_results = []
        all_true_labels = []
        all_pred_labels = []
        class_correct = {}
        class_total = {}
        similarity_scores = []
        unknown_count = 0
        
        # Process batches
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Generate similarity distribution plots for each sample
            '''
            for idx, result in enumerate(batch_results):
                self._plot_similarity_distributions(
                    result, 
                    os.path.join(similarity_plots_dir, f"sample_{len(all_results)-len(batch_results)+idx}")
                )
            '''
            # Update metrics
            for result in batch_results:
                true_class = result['true_class']
                pred_class = result['predicted_class']
                is_correct = result['correct']
                similarity = result['similarity']
                
                all_true_labels.append(true_class)
                all_pred_labels.append(pred_class)
                similarity_scores.append(similarity)
                
                if pred_class == "unknown":
                    unknown_count += 1
                
                if true_class not in class_correct:
                    class_correct[true_class] = 0
                    class_total[true_class] = 0
                
                class_total[true_class] += 1
                if is_correct:
                    class_correct[true_class] += 1
        
        # Calculate overall metrics
        total_correct = sum(1 for r in all_results if r['correct'])
        total_samples = len(all_results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Calculate class accuracies
        class_accuracy = {}
        for cls in class_total:
            class_accuracy[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            print(f"Class {cls}: {class_accuracy[cls]:.4f} ({class_correct[cls]}/{class_total[cls]})")
        
        # Calculate confidence bins
        confidence_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
        for result in all_results:
            bin_idx = min(int(result['similarity'] * 10), 9)
            confidence_bin = f"{bin_idx * 0.1:.1f}-{(bin_idx + 1) * 0.1:.1f}"
            confidence_bins[confidence_bin]['total'] += 1
            if result['correct']:
                confidence_bins[confidence_bin]['correct'] += 1
        
        # Calculate bin accuracies
        for bin_name, bin_data in confidence_bins.items():
            if bin_data['total'] > 0:
                bin_data['accuracy'] = bin_data['correct'] / bin_data['total']
            else:
                bin_data['accuracy'] = 0
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        class_names = sorted(list(set(all_true_labels)))
        pred_class_names = sorted(list(set(all_pred_labels)))
        
        # Make sure "unknown" is in the labels if it appears in predictions
        if "unknown" in pred_class_names and "unknown" not in class_names:
            pred_class_names = [c for c in pred_class_names if c != "unknown"] + ["unknown"]
        
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=class_names)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted'
        )
        
        # Compile results
        evaluation_results = {
            'accuracy': overall_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_accuracy': class_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'unknown_count': unknown_count,
            'unknown_percentage': unknown_count / total_samples if total_samples > 0 else 0,
            'confusion_matrix': cm.tolist(),
            'confidence_bins': {k: v for k, v in confidence_bins.items()},
            'execution_time': time.time() - start_time,
            'mean_similarity': np.mean(similarity_scores),
            'similarity_threshold': self.similarity_threshold
        }
        
        # Create visualizations
        self._create_visualizations(all_results, class_accuracy, cm, class_names, confidence_bins, output_dir)
        
        # Save detailed predictions
        predictions_file = os.path.join(output_dir, 'detailed_predictions.json')
        with open(predictions_file, 'w') as f:
            json.dump([{
                'path': r['path'],
                'true_class': r['true_class'],
                'predicted_class': r['predicted_class'],
                'similarity': r['similarity'],
                'correct': r['correct'],
                'top5_similarities': dict(sorted(r['all_similarities'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5])
            } for r in all_results], f, indent=2)
        
        # Save summary metrics
        summary_file = os.path.join(output_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Print summary
        print("\n===== Evaluation Results =====")
        print(f"Total images: {total_samples}")
        print(f"Correct predictions: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Unknown predictions: {unknown_count} ({evaluation_results['unknown_percentage']:.2%})")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Execution time: {evaluation_results['execution_time']:.2f} seconds")
        
        return evaluation_results
        
    def _plot_similarity_distributions(self, result, output_path):
        """Create plots showing the distribution of similarity scores for each class"""
        full_similarities = result['full_similarities']
        true_class = result['true_class']
        predicted_class = result['predicted_class']
        img_path = result['path']
        
        # Get the filename from the path
        img_filename = os.path.basename(img_path)
        
        # Calculate the number of rows and columns for subplots
        n_classes = len(full_similarities)
        n_cols = 3  # Number of columns in the subplot grid
        n_rows = (n_classes + n_cols - 1) // n_cols  # Ceiling division for number of rows
        
        # Create the figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        
        # Flatten axes array for easier indexing if there are multiple rows
        if n_rows > 1:
            axes = axes.flatten()
        
        # Plot similarity distributions for each class
        for i, (class_name, similarities) in enumerate(sorted(full_similarities.items())):
            ax = axes[i] if n_rows > 1 or n_cols > 1 else axes
            
            # Plot the distribution
            sns.histplot(similarities, kde=True, ax=ax, color='blue' if class_name != predicted_class else 'green')
            
            # Mark the max similarity
            max_sim = max(similarities)
            ax.axvline(max_sim, color='red', linestyle='--', 
                      label=f'Max: {max_sim:.3f}')
            
            # Mark the similarity threshold
            ax.axvline(self.similarity_threshold, color='black', linestyle='-', 
                      label=f'Threshold: {self.similarity_threshold:.3f}')
            
            # Set title and labels
            title = f"{class_name}"
            if class_name == true_class:
                title += " (True Class)"
            if class_name == predicted_class:
                title += " (Predicted)"
                
            ax.set_title(title)
            ax.set_xlabel('Similarity Score')
            ax.set_ylabel('Count')
            ax.legend()
            
            # Set x-axis limits for consistency
            ax.set_xlim(0, 1)
        
        # If there are empty subplots, hide them
        for i in range(len(full_similarities), n_rows * n_cols):
            if n_rows > 1 or n_cols > 1:
                axes[i].axis('off')
        
        # Add a main title with the image info
        plt.suptitle(f"Similarity Distributions - {img_filename}\nTrue: {true_class}, Predicted: {predicted_class}", 
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        # Save the figure
        plt.savefig(f"{output_path}.png")
        plt.close(fig)
        
    def _create_visualizations(self, results, class_accuracy, confusion_matrix, class_names, confidence_bins, output_dir):
        """Create enhanced visualizations for evaluation results"""
        # 1. Confusion matrix with normalized values
        plt.figure(figsize=(12, 10))
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # 2. Class accuracy bar plot
        plt.figure(figsize=(14, 6))
        sorted_classes = sorted(class_accuracy.keys(), key=lambda x: class_accuracy[x], reverse=True)
        accuracies = [class_accuracy[c] for c in sorted_classes]
        
        # Calculate error margin (95% confidence interval)
        class_totals = {cls: sum(1 for item in results if item['true_class'] == cls)
                      for cls in sorted_classes}
        error_margins = [1.96 * np.sqrt((acc * (1 - acc)) / class_totals[cls]) 
                        for acc, cls in zip(accuracies, sorted_classes)]
        
        plt.bar(range(len(sorted_classes)), accuracies, yerr=error_margins)
        plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=90)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy with 95% Confidence Intervals')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_accuracy.png'))
        
        # 3. Similarity distribution plot
        plt.figure(figsize=(10, 6))
        correct_predictions = [r['similarity'] for r in results if r['correct']]
        incorrect_predictions = [r['similarity'] for r in results if not r['correct']]
        unknown_predictions = [r['similarity'] for r in results if r['predicted_class'] == 'unknown']
        
        plt.hist([correct_predictions, incorrect_predictions], bins=20, alpha=0.7, 
                label=['Correct', 'Incorrect'])
        plt.axvline(np.mean(correct_predictions), color='g', linestyle='--', label=f'Mean correct: {np.mean(correct_predictions):.3f}')
        if incorrect_predictions:  # Only add this line if there are incorrect predictions
            plt.axvline(np.mean(incorrect_predictions), color='r', linestyle='--', label=f'Mean incorrect: {np.mean(incorrect_predictions):.3f}')
        plt.axvline(self.similarity_threshold, color='black', linestyle='-', label=f'Threshold: {self.similarity_threshold:.3f}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.title('Distribution of Similarity Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
        
        # 4. Reliability diagram (calibration curve)
        bin_accuracies = []
        bin_confidences = []
        bin_sizes = []
        
        # Sort bins by confidence
        sorted_bins = sorted(confidence_bins.items(), key=lambda x: float(x[0].split('-')[0]))
        
        for bin_name, bin_data in sorted_bins:
            if bin_data['total'] > 0:
                bin_accuracies.append(bin_data['accuracy'])
                # Use the midpoint of the bin as the confidence
                bin_mid = (float(bin_name.split('-')[0]) + float(bin_name.split('-')[1])) / 2
                bin_confidences.append(bin_mid)
                bin_sizes.append(bin_data['total'])
        
        plt.figure(figsize=(10, 6))
        
        # Plot identity line (perfectly calibrated)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Plot calibration curve
        plt.scatter(bin_confidences, bin_accuracies, s=[x/5 for x in bin_sizes], alpha=0.8)
        plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model calibration')
        
        # Add bin counts as text
        for i, (x, y, size) in enumerate(zip(bin_confidences, bin_accuracies, bin_sizes)):
            plt.text(x, y, f"{size}", fontsize=9, ha='center', va='bottom')
        
        # Add threshold line
        plt.axvline(self.similarity_threshold, color='red', linestyle='--', 
                    label=f'Unknown threshold: {self.similarity_threshold}')
        
        plt.xlabel('Confidence (Predicted Similarity)')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram (Calibration Curve)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
        
        # 5. Most confused pairs visualization
        class_confusion = {}
        for i, true_class in enumerate(class_names):
            for j, pred_class in enumerate(class_names):
                if i != j and confusion_matrix[i, j] > 0:
                    class_confusion[(true_class, pred_class)] = confusion_matrix[i, j]
        
        # Get top confused pairs
        top_confused = sorted(class_confusion.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_confused:  # Only create this visualization if there are confused pairs
            plt.figure(figsize=(10, 6))
            pair_labels = [f"{true} → {pred}" for (true, pred), _ in top_confused]
            confusion_counts = [count for _, count in top_confused]
            
            plt.barh(range(len(pair_labels)), confusion_counts, color='salmon')
            plt.yticks(range(len(pair_labels)), pair_labels)
            plt.xlabel('Count')
            plt.ylabel('True → Predicted')
            plt.title('Top Most Confused Class Pairs')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'most_confused_pairs.png'))
            
        # 6. Unknown class analysis
        unknown_preds = [r for r in results if r['predicted_class'] == 'unknown']
        if unknown_preds:
            # Distribution of true classes in unknown predictions
            unknown_true_classes = {}
            for r in unknown_preds:
                true_class = r['true_class']
                if true_class not in unknown_true_classes:
                    unknown_true_classes[true_class] = 0
                unknown_true_classes[true_class] += 1
            
            # Sort by frequency
            sorted_unknown = sorted(unknown_true_classes.items(), key=lambda x: x[1], reverse=True)
            
            plt.figure(figsize=(10, 6))
            class_labels = [cls for cls, _ in sorted_unknown]
            class_counts = [count for _, count in sorted_unknown]
            
            plt.barh(range(len(class_labels)), class_counts, color='lightblue')
            plt.yticks(range(len(class_labels)), class_labels)
            plt.xlabel('Count')
            plt.ylabel('True Class')
            plt.title('Distribution of True Classes in Unknown Predictions')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'unknown_class_distribution.png'))

def main():
    parser = argparse.ArgumentParser(description='Batch Inference for Fish Classification Model')
    parser.add_argument('--val_dir', type=str, default="../../data/fish-cl-dataset/val/",
                        help='Path to validation data directory with class subdirectories')
    parser.add_argument('--model_path', type=str, default='models/fish_classifier_ddp.pth',
                        help='Path to trained model')
    parser.add_argument('--reference_dir', type=str, default="data/ensemble_templates/",
                        help='Path to directory containing reference images for each class')
    parser.add_argument('--output_dir', type=str, default='results/batch',
                        help='Path to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                        help='Threshold for unknown class prediction')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    
    # Check if reference directory exists
    if not os.path.exists(args.reference_dir):
        print(f"Reference directory not found at {args.reference_dir}")
        return
        
    # Initialize batch inference model
    batch_model = BatchInferenceModel(
        model_path=args.model_path,
        reference_dir=args.reference_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        similarity_threshold=args.similarity_threshold
    )
    
    # Run evaluation
    batch_model.evaluate(
        data_dir=args.val_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 