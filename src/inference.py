"""
Inference for ImSecu2026 Challenge

Team: Antonello Di Pede, Dario Gosmar, Andrea Gaudino
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models import CLIPDetector, DINOv2Detector, TransFusionNet

import warnings
warnings.filterwarnings('ignore')


class TestDataset(Dataset):
    """Load test images without labels"""
    
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        self.image_names = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        print(f"Loaded {len(self.image_names)} test images")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = np.zeros((448, 448, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, img_name


def predict(model_path, model_type, test_images_dir, output_csv, device='mps', batch_size=16):
    """
    Generate predictions on test set
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model (clip, dino or transfusion)
        test_images_dir: Directory with test images
        output_csv: Output CSV path
        device: Device (mps/cuda/cpu)
        batch_size: Batch size
    """
    
    # Load model
    if model_type == 'clip':
        model = CLIPDetector()
    elif model_type == 'dino':
        model = DINOv2Detector()
    elif model_type == 'transfusion':
        model = TransFusionNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {model_path}")
    if 'auc' in checkpoint:
        print(f"  Validation AUC: {checkpoint['auc']:.4f}")
    
    # Test transform
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=512, interpolation=1),
        A.CenterCrop(448, 448),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load dataset
    test_dataset = TestDataset(test_images_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Predict
    results = []
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for img_name, prob in zip(img_names, probs):
                results.append({
                    'image_name': img_name,
                    'label': float(prob)
                })
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Stats
    print(f"\n✓ Predictions saved: {output_csv}")
    print(f"\nPrediction Statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Mean prediction: {df['label'].mean():.4f}")
    print(f"  Std prediction: {df['label'].std():.4f}")
    print(f"  Min: {df['label'].min():.4f} | Max: {df['label'].max():.4f}")
    print(f"  Predicted Fake (>0.5): {(df['label'] > 0.5).sum()} ({(df['label'] > 0.5).sum()/len(df)*100:.1f}%)")
    print(f"  Predicted Real (≤0.5): {(df['label'] <= 0.5).sum()} ({(df['label'] <= 0.5).sum()/len(df)*100:.1f}%)")
    
    return df