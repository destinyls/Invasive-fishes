import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from model import FishClassifier
from dataset import FishDataset

def main():
    parser = argparse.ArgumentParser(description='Fish Classification Demo')
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict'],
                        help='Mode: train or predict')
    parser.add_argument('--data_dir', type=str, default='../../data/fish-clip-dataset',
                        help='Path to fish clip dataset directory')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image for prediction (only in predict mode)')
    parser.add_argument('--model_path', type=str, default='models/fish_classifier.pth',
                        help='Path to model for prediction or to save trained model')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet101', 'efficientnet_b2'],
                       help='Backbone network architecture')
    parser.add_argument('--reference_dir', type=str, default=None,
                       help='Directory containing reference images for each class')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_demo(args.data_dir, args.model_path, args.backbone)
    else:
        if args.reference_dir is None:
            args.reference_dir = os.path.join(args.data_dir, 'train')
            print(f"No reference directory specified, using training directory: {args.reference_dir}")
        predict_demo(args.image, args.model_path, args.reference_dir)

def train_demo(data_dir, model_path, backbone='resnet50'):
    """Demo for training the fish classifier with BYOL"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory '{train_dir}' does not exist.")
        print("Please make sure the dataset path is correct.")
        return

    # Setup dataloader
    train_loader, val_loader = FishDataset.get_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32,
        image_size=224
    )
    
    # Initialize model
    model = FishClassifier(backbone=backbone)
    print(f"Using {backbone} as backbone network")
    
    # Train the model
    print("Training model with BYOL self-supervised learning...")
    _train_byol(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Reduced for demo
        lr=1e-4,
        weight_decay=1e-6,
        save_path=model_path
    )
    
    print(f"Demo training completed. Model saved to {model_path}")
    print(f"You can now use this model for prediction with --mode predict --image path/to/image.jpg")

def _train_byol(model, train_loader, val_loader, epochs, lr, weight_decay, save_path):
    """Training function using standard BYOL"""
    # Setup optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.online_backbone.parameters(), 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': model.online_projector.parameters()},
        {'params': model.online_predictor.parameters()}
    ], lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_data in train_loader:
            src_images, _ = batch_data
            src_images = src_images.to(model.device)
            
            # Forward pass with standard BYOL
            optimizer.zero_grad()
            loss = model(src_images)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update target network
            model.update_target_network()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_count}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_loss / batch_count
        print(f'Epoch: {epoch+1}/{epochs}, Average Train Loss: {avg_train_loss:.4f}')
        
        # Validation phase
        if val_loader is not None:
            val_loss = validate_byol(model, val_loader)
            print(f'Epoch: {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        else:
            # Save model periodically if no validation
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch+1}")
        
        # Step the scheduler
        scheduler.step()

def validate_byol(model, val_loader):
    """Validate the BYOL model"""
    model.eval()
    val_loss = 0
    val_batch_count = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            src_images, _ = batch_data
            src_images = src_images.to(model.device)
            
            # Standard BYOL forward pass
            loss = model(src_images)
            
            val_loss += loss.item()
            val_batch_count += 1
    
    return val_loss / val_batch_count

def predict_demo(image_path, model_path, reference_dir):
    """Demo for predicting with the fish classifier"""
    if image_path is None:
        print("Error: Please provide an image path with --image")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        print("Please train the model first with --mode train")
        return
    
    if not os.path.exists(reference_dir):
        print(f"Error: Reference directory '{reference_dir}' does not exist.")
        return
    
    # Initialize model
    model = FishClassifier(model_path=model_path)
    
    # Extract reference features
    reference_features = extract_reference_features(model, reference_dir)
    if not reference_features:
        print("No reference features extracted. Cannot proceed with prediction.")
        return
    
    # Make prediction
    prediction = predict_image(model, image_path, reference_features)
    
    # Display results
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot top 5 predictions
    plt.subplot(1, 2, 2)
    sorted_similarities = sorted(prediction['all_similarities'].items(), key=lambda x: x[1], reverse=True)[:5]
    classes = [x[0] for x in sorted_similarities]
    similarities = [x[1] for x in sorted_similarities]
    
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, similarities, color='skyblue')
    plt.yticks(y_pos, classes)
    plt.xlabel('Similarity Score')
    plt.title(f'Predicted: {prediction["predicted_class"]}')
    
    # Save plot
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, 'prediction_result.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    
    print(f"\nPrediction results for {image_path}:")
    print(f"  Predicted class: {prediction['predicted_class']}")
    print(f"  Similarity score: {prediction['similarity']:.4f}")
    print("\nTop similarities:")
    for class_name, similarity in sorted_similarities:
        print(f"  {class_name}: {similarity:.4f}")
    
    print(f"\nVisualization saved to {output_path}")

def extract_reference_features(model, reference_dir):
    """Extract features from reference images for each class"""
    print(f"Extracting reference features from {reference_dir}")
    reference_features = {}
    
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
        for img_file in images[:5]:  # Use up to 5 images per class for demo
            image_path = os.path.join(class_dir, img_file)
            try:
                features = model.extract_features(image_path)
                features = F.normalize(features, dim=1)
                class_features.append(features)
            except Exception as e:
                print(f"Error extracting features from {image_path}: {e}")
        
        if class_features:
            # Compute average feature for this class
            class_features = torch.cat(class_features, dim=0)
            avg_feature = torch.mean(class_features, dim=0, keepdim=True)
            reference_features[class_name] = F.normalize(avg_feature, dim=1)
            print(f"Extracted features for class {class_name} from {len(class_features)} images")
    
    print(f"Extracted reference features for {len(reference_features)} classes")
    return reference_features

def predict_image(model, image_path, reference_features):
    """Predict the class of an input image by comparing with reference features"""
    # Extract features from the input image
    input_feature = model.extract_features(image_path)
    input_feature = F.normalize(input_feature, dim=1)
    
    # Compare with reference features
    similarities = {}
    for class_name, ref_feature in reference_features.items():
        similarity = F.cosine_similarity(input_feature, ref_feature)
        similarities[class_name] = similarity.item()
    
    # Find the most similar class
    predicted_class = max(similarities.items(), key=lambda x: x[1])
    
    return {
        'predicted_class': predicted_class[0],
        'similarity': predicted_class[1],
        'all_similarities': similarities
    }

if __name__ == "__main__":
    main() 