"""
Training pipeline for ImSecu2026 Challenge

Team: Antonello Di Pede, Dario Gosmar, Andrea Gaudino
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings('ignore')

def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=2):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (mps/cuda/cpu)
        accumulation_steps: Gradient accumulation steps
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Validate model and compute ROC AUC
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        device: Device
        
    Returns:
        ROC AUC score
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    return auc


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, 
                device='mps', save_path='checkpoints/best_model.pth', 
                accumulation_steps=2):
    """
    Full training pipeline with history tracking
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    history = {
        'train_loss': [],
        'val_auc': [],
        'epochs': []
    }
    
    print("=" * 60)
    print("Training started")
    print("=" * 60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                device, accumulation_steps)
        
        # Validate
        val_auc = validate(model, val_loader, device)
        
        # Learning rate step
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['epochs'].append(epoch + 1)
        
        print(f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': val_auc,
                'history': history
            }, save_path)
            print(f"✓ Best model saved (AUC: {val_auc:.4f})")
    
    # Save final history
    history_path = save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_training_curves(history, save_path.replace('.pth', '_curves.png'))
    
    print("\n" + "=" * 60)
    print(f"Training completed! Best AUC: {best_auc:.4f}")
    print("=" * 60)
    
    return model, history


def plot_training_curves(history, save_path):
    """
    Plot training loss and validation AUC curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = history['epochs']
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # AUC curve
    ax2.plot(epochs, history['val_auc'], 'g-', linewidth=2, label='Val AUC')
    ax2.axhline(y=max(history['val_auc']), color='r', linestyle='--', 
                alpha=0.5, label=f'Best: {max(history["val_auc"]):.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved: {save_path}")