"""
Dataset loader for NTIRE 2026 Robust AI-Generated Image Detection
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class NTIRETrainDataset(Dataset):
    """
    Load training images from shards with labels.csv
    
    Args:
        shard_dir: Base directory containing shard_0, shard_1, ...
        shard_nums: List of shard indices to use (None = all 6 shards)
        transform: Albumentations transform pipeline
    """
    
    def __init__(self, shard_dir, shard_nums=None, transform=None):
        self.shard_dir = shard_dir
        self.transform = transform
        
        # Select shard directories
        if shard_nums is None:
            shard_nums = list(range(6))  # All shards
        
        self.shard_dirs = [
            os.path.join(shard_dir, f'shard_{i}') 
            for i in shard_nums 
            if os.path.isdir(os.path.join(shard_dir, f'shard_{i}'))
        ]
        
        # Load and concatenate all labels.csv
        label_dfs = []
        for shard_path in self.shard_dirs:
            csv_path = os.path.join(shard_path, 'labels.csv')
            df = pd.read_csv(csv_path)
            df['shard_path'] = shard_path
            label_dfs.append(df)
        
        self.label_df = pd.concat(label_dfs, ignore_index=True)
        
        # Print dataset info
        n_real = (self.label_df['label'] == 0).sum()
        n_fake = (self.label_df['label'] == 1).sum()
        print(f"Loaded {len(self.shard_dirs)} shards: {len(self.label_df)} images")
        print(f"  Real: {n_real} | Fake: {n_fake}")
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        
        # Build path to image
        img_path = os.path.join(row['shard_path'], 'images', row['image_name'])
        
        # Load image as RGB numpy array
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = np.zeros((504, 504, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = int(row['label'])
        return image, label
    
def get_transforms():
    """
    ULTRA-AGGRESSIVE augmentation for maximum robustness
    100% aggressivity - compensated for missing ISONoise
    """
    
    train_transform = A.Compose([
        # Initial resize
        A.SmallestMaxSize(max_size=512, interpolation=1),
        A.RandomCrop(448, 448),
        A.HorizontalFlip(p=0.5),
        
        # ============================================
        # LEVEL 1: EXTREME COMPRESSION & BLUR (98%)
        # ============================================
        A.OneOf([
            # JPEG compression: VERY LOW quality
            A.ImageCompression(
                quality_lower=10,
                quality_upper=95,
                p=1.0
            ),
            
            # Gaussian blur: VERY STRONG
            A.GaussianBlur(
                blur_limit=(3, 21),
                sigma_limit=(0.1, 3.0),
                p=1.0
            ),
            
            # Motion blur: simulates camera shake
            A.MotionBlur(
                blur_limit=(5, 19),
                p=1.0
            ),
            
            # Median blur: different blur type
            A.MedianBlur(
                blur_limit=(3, 11),
                p=1.0
            ),
        ], p=0.98),
        
        # ============================================
        # LEVEL 2: EXTREME DOWNSCALING (95%)
        # ============================================
        A.Downscale(
            scale_min=0.1,
            scale_max=0.7,
            interpolation=0,
            p=0.95
        ),
        
        # ============================================
        # LEVEL 3: NOISE & ARTIFACTS (95% - COMPENSATED!)
        # ============================================
        A.OneOf([
            # Gaussian noise: VERY HIGH variance
            A.GaussNoise(
                var_limit=(40.0, 150.0),  # Increased to compensate ISONoise
                mean=0,
                per_channel=True,
                p=1.0
            ),
            
            # Multiplicative noise: STRONGER
            A.MultiplicativeNoise(
                multiplier=(0.6, 1.4),    # Wider range
                per_channel=True,
                elementwise=True,
                p=1.0
            ),
            
            # Gaussian noise with bias (simulates sensor noise)
            A.GaussNoise(
                var_limit=(50.0, 200.0),  # Extreme noise
                mean=10,
                per_channel=True,
                p=1.0
            ),
        ], p=0.95),  # Was 0.90, now 0.95!
        
        # ============================================
        # LEVEL 4: COLOR DEGRADATION (85%)
        # ============================================
        A.OneOf([
            # Posterize: reduce color depth
            A.Posterize(
                num_bits=(2, 6),
                p=1.0
            ),
            
            # Color jitter: EXTREME shifts
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2,
                p=1.0
            ),
            
            # Random gamma: lighting changes
            A.RandomGamma(
                gamma_limit=(40, 160),
                p=1.0
            ),
            
            # Random brightness/contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=1.0
            ),
        ], p=0.85),
        
        # ============================================
        # LEVEL 5: CASCADE - MULTIPLE DEGRADATIONS (85%)
        # ============================================
        A.Sequential([
            # JPEG + Blur together
            A.ImageCompression(
                quality_lower=15,
                quality_upper=70,
                p=0.7
            ),
            A.GaussianBlur(
                blur_limit=(3, 13),
                p=0.6
            ),
            # Add noise on top
            A.GaussNoise(
                var_limit=(30.0, 100.0),
                per_channel=True,
                p=0.5
            ),
        ], p=0.85),  # Was 0.80, now 0.85!
        
        # ============================================
        # LEVEL 6: SPATIAL DISTORTIONS (30%)
        # ============================================
        A.OneOf([
            # Grid distortion
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=1.0
            ),
            
            # Optical distortion (lens effects)
            A.OpticalDistortion(
                distort_limit=0.3,
                shift_limit=0.3,
                p=1.0
            ),
        ], p=0.30),
        
        # ============================================
        # FINAL: Normalize
        # ============================================
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # ============================================
    # VALIDATION: MODERATE DEGRADATION (70%)
    # ============================================
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=512, interpolation=1),
        A.CenterCrop(448, 448),
        
        # Moderate degradations for validation
        A.OneOf([
            A.ImageCompression(
                quality_lower=40,
                quality_upper=90,
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(3, 11),
                p=1.0
            ),
            A.Downscale(
                scale_min=0.3,
                scale_max=0.7,
                interpolation=0,
                p=1.0
            ),
        ], p=0.70),
        
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

class TransformSubset(Dataset):
    """
    Apply specific transform to subset of dataset
    
    Allows using different transforms for train/val from same base dataset
    """
    
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get image from base dataset (without transform)
        actual_idx = self.indices[idx]
        row = self.base_dataset.label_df.iloc[actual_idx]
        
        # Build path and load image
        img_path = os.path.join(row['shard_path'], 'images', row['image_name'])
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = np.zeros((504, 504, 3), dtype=np.uint8)
        
        # Apply THIS subset's transform
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = int(row['label'])
        return image, label
