import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict


class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=224, return_paths=False):
        """
        Args:
            root_dir (string): Directory with all the fish images organized in class folders
            transform (callable, optional): Optional transform to be applied on samples
            image_size (int): Size to resize images to
            return_paths (bool): Whether to return image paths along with images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.return_paths = return_paths
        self.classes = []
        self.samples = []
        
        # If no transform provided, create a default one
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Load all image paths and their labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx, class_name))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        
        # Load the source image
        src_img = Image.open(img_path).convert('RGB')
        
        # Transform the source image
        if self.transform:
            src_tensor = self.transform(src_img)
        else:
            src_tensor = transforms.ToTensor()(src_img)
        
        # Return image and class index
        if self.return_paths:
            return src_tensor, class_idx, img_path
        else:
            return src_tensor, class_idx
    
    @staticmethod
    def get_data_loaders(train_dir, val_dir=None, batch_size=32, image_size=224, 
                         num_workers=4, return_paths=False):
        """
        Creates train and validation dataloaders
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data, if None, returns None for val_loader
            batch_size: Batch size for dataloaders
            image_size: Size to resize images to
            num_workers: Number of workers for DataLoader
            return_paths: Whether to return image paths
            
        Returns:
            train_loader, val_loader (None if val_dir is None)
        """
        # Create training data loader
        train_dataset = FishDataset(
            root_dir=train_dir,
            image_size=image_size,
            return_paths=return_paths
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Create validation data loader if val_dir provided
        val_loader = None
        if val_dir and os.path.exists(val_dir):
            val_dataset = FishDataset(
                root_dir=val_dir,
                image_size=image_size,
                return_paths=return_paths
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return train_loader, val_loader

    @staticmethod
    def get_datasets(train_dir, val_dir=None, image_size=224, return_paths=False):
        """
        Creates train and validation datasets (for distributed training)
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data, if None, returns None for val_dataset
            image_size: Size to resize images to
            return_paths: Whether to return image paths
            
        Returns:
            train_dataset, val_dataset (None if val_dir is None)
        """
        # Create training dataset
        train_dataset = FishDataset(
            root_dir=train_dir,
            image_size=image_size,
            return_paths=return_paths
        )
        
        # Create validation dataset if val_dir provided
        val_dataset = None
        if val_dir and os.path.exists(val_dir):
            val_dataset = FishDataset(
                root_dir=val_dir,
                image_size=image_size,
                return_paths=return_paths
            )
        
        return train_dataset, val_dataset

class ClassPairFishDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=224, return_paths=False):
        """
        Dataset that returns pairs of different images from the same class.
        
        Args:
            root_dir (string): Directory with all the fish images organized in class folders
            transform (callable, optional): Optional transform to be applied on samples
            image_size (int): Size to resize images to
            return_paths (bool): Whether to return image paths along with images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.return_paths = return_paths
        self.classes = []
        self.samples = []
        self.class_to_indices = defaultdict(list)
        
        # If no transform provided, create a default one
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Load all image paths and their labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                class_images = []
                
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx, class_name))
                        # Store index by class
                        self.class_to_indices[class_idx].append(len(self.samples) - 1)
        
        # Filter out classes with only one image
        valid_classes = [cls for cls, indices in self.class_to_indices.items() if len(indices) > 1]
        if len(valid_classes) < len(self.class_to_indices):
            print(f"Warning: Removed {len(self.class_to_indices) - len(valid_classes)} classes with only one image")
            # Keep only samples from valid classes
            self.samples = [sample for sample in self.samples if sample[1] in valid_classes]
            # Rebuild class_to_indices
            self.class_to_indices = defaultdict(list)
            for idx, (_, class_idx, _) in enumerate(self.samples):
                self.class_to_indices[class_idx].append(idx)
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_to_indices)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        
        # Find another image from the same class (but not the same image)
        same_class_indices = self.class_to_indices[class_idx]
        pair_candidates = [i for i in same_class_indices if i != idx]
        
        if not pair_candidates:
            # This shouldn't happen due to our filtering, but just in case
            pair_idx = idx  # Fall back to using the same image
        else:
            pair_idx = random.choice(pair_candidates)
        
        pair_img_path, _, _ = self.samples[pair_idx]
        
        # Load images without converting to RGB to keep the mask channel
        anchor_img = Image.open(img_path)
        pair_img = Image.open(pair_img_path)
        
        # Process each image to remove background using the mask
        anchor_img = self._remove_background(anchor_img)
        pair_img = self._remove_background(pair_img)
        
        # Transform the images
        if self.transform:
            anchor_tensor = self.transform(anchor_img)
            pair_tensor = self.transform(pair_img)
        else:
            anchor_tensor = transforms.ToTensor()(anchor_img)
            pair_tensor = transforms.ToTensor()(pair_img)
        
        # Return image pair and class index
        if self.return_paths:
            print(f"Returning image pair and class index: {class_idx}, {img_path}, {pair_img_path}")
            return anchor_tensor, pair_tensor, class_idx, img_path, pair_img_path
        else:
            return anchor_tensor, pair_tensor, class_idx
    
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
        if img.mode == 'RGBA' or (hasattr(img, 'n_frames') and img.n_frames == 4):
            # For webp with 4 channels, split the channels
            r, g, b, mask = img.split()
                        
            # Apply mask to keep only foreground
            rgb_masked = Image.new("RGB", img.size, (0, 0, 0))
            rgb_masked.paste(img.convert('RGB'), mask=mask)
            
            return rgb_masked
        else:
            # If not a 4-channel image, just convert to RGB
            return img.convert('RGB')
    
    @staticmethod
    def get_data_loaders(train_dir, val_dir=None, batch_size=32, image_size=224, 
                         num_workers=4, return_paths=False):
        """
        Creates train and validation dataloaders with class-paired samples
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data, if None, returns None for val_loader
            batch_size: Batch size for dataloaders
            image_size: Size to resize images to
            num_workers: Number of workers for DataLoader
            return_paths: Whether to return image paths
            
        Returns:
            train_loader, val_loader (None if val_dir is None)
        """
        # Create training data loader
        train_dataset = ClassPairFishDataset(
            root_dir=train_dir,
            image_size=image_size,
            return_paths=return_paths
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Create validation data loader if val_dir provided
        val_loader = None
        if val_dir and os.path.exists(val_dir):
            val_dataset = ClassPairFishDataset(
                root_dir=val_dir,
                image_size=image_size,
                return_paths=return_paths
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return train_loader, val_loader

    @staticmethod
    def get_datasets(train_dir, val_dir=None, image_size=224, return_paths=False):
        """
        Creates train and validation datasets with class-paired samples (for distributed training)
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data, if None, returns None for val_dataset
            image_size: Size to resize images to
            return_paths: Whether to return image paths
            
        Returns:
            train_dataset, val_dataset (None if val_dir is None)
        """
        # Create training dataset
        train_dataset = ClassPairFishDataset(
            root_dir=train_dir,
            image_size=image_size,
            return_paths=return_paths
        )
        
        # Create validation dataset if val_dir provided
        val_dataset = None
        if val_dir and os.path.exists(val_dir):
            val_dataset = ClassPairFishDataset(
                root_dir=val_dir,
                image_size=image_size,
                return_paths=return_paths
            )
        
        return train_dataset, val_dataset 