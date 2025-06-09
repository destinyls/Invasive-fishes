import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import copy
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR

class EMA:
    """Exponential Moving Average for target network update"""
    def __init__(self, beta=0.996):  # Increased from 0.99 for slower, more stable updates
        self.beta = beta
        
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, target_network, online_network):
    """Update target network parameters using EMA"""
    for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
        old_weight, new_weight = target_param.data, online_param.data
        target_param.data = ema_updater.update_average(old_weight, new_weight)

class ProjectionHead(nn.Module):
    """Improved MLP projection head with 3 layers"""
    def __init__(self, input_dim, hidden_dim=4096, output_dim=512):  # Increased output dimension from 256 to 512
        super().__init__()
        self.net = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # Second layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # Final layer
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class PredictionHead(nn.Module):
    """Prediction head for BYOL"""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TensorAugmentation(nn.Module):
    """Enhanced augmentation for tensor inputs"""
    def __init__(self, image_size=224, stronger_aug=False):
        super().__init__()
        self.image_size = image_size
        self.stronger_aug = stronger_aug
        
    def forward(self, x):
        # Input is (C, H, W) tensor
        # Apply random crop with variable scale based on stronger_aug
        scale = (0.08, 1.0) if self.stronger_aug else (0.2, 1.0)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x, scale=scale, ratio=(0.75, 1.33))
        x = TF.resize(TF.crop(x, i, j, h, w), [self.image_size, self.image_size])
        
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            x = TF.hflip(x)
            
        # Random vertical flip (only in stronger augmentation mode)
        if self.stronger_aug and torch.rand(1).item() < 0.3:
            x = TF.vflip(x)
        
        # Color jitter with stronger parameters when stronger_aug=True
        if torch.rand(1).item() < 0.8:
            strength = 0.5 if self.stronger_aug else 0.2
            brightness_factor = 1.0 + strength * (torch.rand(1).item() * 2 - 1)
            contrast_factor = 1.0 + strength * (torch.rand(1).item() * 2 - 1)
            saturation_factor = 1.0 + strength * (torch.rand(1).item() * 2 - 1)
            
            x = TF.adjust_brightness(x, brightness_factor)
            x = TF.adjust_contrast(x, contrast_factor)
            x = TF.adjust_saturation(x, saturation_factor)
        
        # Random grayscale with increased probability in stronger mode
        grayscale_prob = 0.4 if self.stronger_aug else 0.2
        if torch.rand(1).item() < grayscale_prob:
            x = TF.rgb_to_grayscale(x, num_output_channels=3)
            
        return x

class FishClassifier(nn.Module):
    def __init__(self, model_path=None, image_size=224, backbone="resnet50"):
        super().__init__()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize backbone model
        self.backbone_name = backbone
        
        if self.backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == "resnet101":
            self.backbone = models.resnet101(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=True)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'resnet50', 'resnet101', 'efficientnet_b2'")
        
        # Output dimensions for projection and prediction
        self.projection_dim = 512  # Increased from 256
        
        # Create online encoder components with improved heads
        self.online_backbone = self.backbone
        self.online_projector = ProjectionHead(self.feature_dim, hidden_dim=4096, output_dim=self.projection_dim)
        self.online_predictor = PredictionHead(self.projection_dim, hidden_dim=1024, output_dim=self.projection_dim)
        
        # Create target encoder components (no gradient)
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Stop gradient for target network
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        # Moving average updater with higher momentum
        self.ema_updater = EMA(beta=0.996)
        
        # Move to device
        self.to(self.device)
        
        # Transform for data augmentation (for PIL images)
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)  # Stronger color jitter
            ], p=0.8),
            transforms.RandomGrayscale(p=0.3),  # Increased probability
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Tensor augmentation for when input is already a tensor
        self.tensor_augmentation1 = TensorAugmentation(image_size, stronger_aug=False)
        self.tensor_augmentation2 = TensorAugmentation(image_size, stronger_aug=True)  # Stronger for second view
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                print(f"Loaded model from {model_path}")
                self.load_state_dict(state_dict['model_state_dict'], strict=False)
            else:
                print(f"Loaded model from {model_path}")
                self.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {model_path}")
    
    def _extract_backbone_features(self, x):
        """Extract features from backbone"""
        return self.online_backbone(x)
    
    def forward(self, x1, x2=None):
        """
        Forward pass for training using BYOL approach with class-paired images
        
        Args:
            x1: First batch of images (shape [B, C, H, W])
            x2: Second batch of images from same class (shape [B, C, H, W]),
                if None, uses standard BYOL with augmentations of x1
        
        Returns:
            loss: BYOL loss value
        """
        batch_size = x1.shape[0]
        
        if x2 is None:
            # Fallback to standard BYOL with two augmentations of the same image
            # Create two augmented views of each image using different augmentation strengths
            x1_aug = torch.stack([self.tensor_augmentation1(img) for img in x1])
            x2_aug = torch.stack([self.tensor_augmentation2(img) for img in x1])
        else:
            # Use class-paired approach with light augmentation on both images
            x1_aug = torch.stack([self.tensor_augmentation1(img) for img in x1])
            x2_aug = torch.stack([self.tensor_augmentation1(img) for img in x2])
        
        # Online network forward passes
        online_feat1 = self.online_backbone(x1_aug)
        online_proj1 = self.online_projector(online_feat1)
        online_pred1 = self.online_predictor(online_proj1)
        
        online_feat2 = self.online_backbone(x2_aug)
        online_proj2 = self.online_projector(online_feat2)
        online_pred2 = self.online_predictor(online_proj2)
        
        # Target network forward passes (no gradients)
        with torch.no_grad():
            target_feat1 = self.target_backbone(x1_aug)
            target_proj1 = self.target_projector(target_feat1)
            
            target_feat2 = self.target_backbone(x2_aug)
            target_proj2 = self.target_projector(target_feat2)
        
        # BYOL loss (symmetric)
        loss1 = self._compute_loss(online_pred1, target_proj2.detach())
        loss2 = self._compute_loss(online_pred2, target_proj1.detach())
        
        loss = (loss1 + loss2) / 2
        return loss
    
    def _compute_loss(self, online, target):
        """
        Classic BYOL loss function - mean squared error between normalized vectors
        """
        online = F.normalize(online, dim=1, p=2)
        target = F.normalize(target, dim=1, p=2)
        
        # Compute MSE loss between normalized vectors (classic BYOL loss)
        loss = 2 - 2 * (online * target).sum(dim=1)
        
        return loss.mean()
    
    def update_target_network(self):
        """Update target network with EMA"""
        update_moving_average(self.ema_updater, self.target_backbone, self.online_backbone)
        update_moving_average(self.ema_updater, self.target_projector, self.online_projector)
    