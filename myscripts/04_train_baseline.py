#!/usr/bin/env python3
"""
MIDOG 2021 Baseline Model Training
ResNet18 with 3-fold LOSO cross-validation
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import logging

# Import config
sys.path.append('/workspace/myscripts')
from config import *

# Setup logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    """DataLoader worker별 고정 seed"""
    np.random.seed(RANDOM_SEED + worker_id)
    torch.manual_seed(RANDOM_SEED + worker_id)


class MitosisDataset(Dataset):
    """MIDOG 패치 데이터셋"""
    
    # Scanner 이름 매핑 (CSV와 폴더명 동일)
    SCANNER_MAP = {
        'Aperio-CS2': 'Aperio-CS2',
        'Hamamatsu-XR': 'Hamamatsu-XR',
        'Hamamatsu-S360': 'Hamamatsu-S360'
    }
    
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.patches_dir = PROCESSED_DIR / 'patches'
        
        logger.info(f"Patches directory: {self.patches_dir}")
        logger.info(f"Directory exists: {self.patches_dir.exists()}")
        
        # List all scanner folders
        if self.patches_dir.exists():
            scanner_folders = [d.name for d in self.patches_dir.iterdir() if d.is_dir()]
            logger.info(f"Scanner folders found: {scanner_folders}")
            
            # Check first scanner folder structure
            if scanner_folders:
                first_scanner = self.patches_dir / scanner_folders[0]
                label_folders = [d.name for d in first_scanner.iterdir() if d.is_dir()]
                logger.info(f"Label folders in {scanner_folders[0]}: {label_folders}")
                
                if label_folders:
                    first_label = first_scanner / label_folders[0]
                    sample_files = list(first_label.glob('*.png'))[:3]
                    logger.info(f"Sample files in {scanner_folders[0]}/{label_folders[0]}: {[f.name for f in sample_files]}")
        
        # Build patch file list from actual files
        self.patch_list = []
        
        for idx, row in self.df.iterrows():
            scanner_folder = self.SCANNER_MAP.get(row['scanner'], row['scanner'])
            file_num = row['file_number']
            
            if idx < 2:  # Log first 2 images
                logger.info(f"Processing image {idx}: file_number={file_num}, scanner={row['scanner']} -> {scanner_folder}")
            
            # Collect all patches for this image
            for label_folder, label_value in [('positive', 1), ('negative', 0)]:
                patch_dir = self.patches_dir / scanner_folder / label_folder
                
                if idx < 2:
                    logger.info(f"  Checking: {patch_dir}, exists={patch_dir.exists()}")
                
                if not patch_dir.exists():
                    if idx < 2:
                        logger.warning(f"  Directory not found: {patch_dir}")
                    continue
                
                # Find all patches for this file number
                pattern = f'{file_num:03d}_*.png'
                matches = list(patch_dir.glob(pattern))
                
                if idx < 2:
                    logger.info(f"  Pattern '{pattern}': found {len(matches)} files")
                    if matches:
                        logger.info(f"  Sample: {matches[0].name}")
                
                for patch_file in matches:
                    self.patch_list.append({
                        'path': patch_file,
                        'label': label_value,
                        'scanner': row['scanner'],
                        'file_number': file_num
                    })
        
        logger.info(f"Found {len(self.patch_list)} patches from {len(self.df)} images")
        
    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, idx):
        patch_info = self.patch_list[idx]
        
        # Load image
        img = Image.open(patch_info['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = patch_info['label']
        
        return img, label


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def build_model():
    """ResNet18 모델 생성"""
    model = models.resnet18(pretrained=True)
    
    # Modify final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """1 epoch 학습"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Store predictions for metrics
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.0
        
    return epoch_loss, auroc


def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate metrics
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except:
        auroc, auprc = 0.0, 0.0
    
    return epoch_loss, auroc, auprc, all_labels, all_probs


def train_fold(fold_num, device):
    """1개 fold 학습"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Fold {fold_num}")
    logger.info(f"{'='*50}")
    
    # Paths
    train_csv = METADATA_DIR / f'fold{fold_num}_train.csv'
    val_csv = METADATA_DIR / f'fold{fold_num}_val.csv'
    test_csv = METADATA_DIR / f'fold{fold_num}_test.csv'
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = MitosisDataset(train_csv, train_transform)
    val_dataset = MitosisDataset(val_csv, val_transform)
    test_dataset = MitosisDataset(test_csv, val_transform)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory'],
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory'],
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory'],
        worker_init_fn=worker_init_fn
    )
    
    # Model
    model = build_model().to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_auroc = 0.0
    patience_counter = 0
    patience = TRAINING_CONFIG['early_stopping_patience']
    
    history = {
        'train_loss': [], 'train_auroc': [],
        'val_loss': [], 'val_auroc': [], 'val_auprc': []
    }
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        
        # Train
        train_loss, train_auroc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auroc, val_auprc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_auroc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_auroc'].append(train_auroc)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_auprc'].append(val_auprc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_counter = 0
            
            model_path = RESULTS_DIR / 'models' / f'fold{fold_num}_best.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_auprc': val_auprc
            }, model_path)
            logger.info(f"Saved best model (AUROC: {best_auroc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    
    # Load best model for test
    checkpoint = torch.load(RESULTS_DIR / 'models' / f'fold{fold_num}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    logger.info("\nEvaluating on test set...")
    test_loss, test_auroc, test_auprc, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
    
    # Save results
    results = {
        'fold': fold_num,
        'best_val_auroc': best_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'epochs_trained': epoch + 1
    }
    
    return results, history, test_labels, test_probs


def main():
    """Main training loop"""
    logger.info("Starting MIDOG Baseline Training")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    # Set device
    device = torch.device(DEVICE)
    
    # Train all folds
    all_results = []
    
    for fold in [1, 2, 3]:
        results, history, test_labels, test_probs = train_fold(fold, device)
        all_results.append(results)
        
        # Save history
        history_df = pd.DataFrame(history)
        history_path = RESULTS_DIR / 'metrics' / f'fold{fold}_history.csv'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_path, index=False)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'label': test_labels,
            'prob': test_probs
        })
        pred_path = RESULTS_DIR / 'metrics' / f'fold{fold}_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
    
    # Summary
    results_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / 'metrics' / 'baseline_summary.csv'
    results_df.to_csv(summary_path, index=False)
    
    logger.info("\n" + "="*50)
    logger.info("Baseline Training Complete")
    logger.info("="*50)
    logger.info(f"\n{results_df}")
    logger.info(f"\nMean Test AUROC: {results_df['test_auroc'].mean():.4f} ± {results_df['test_auroc'].std():.4f}")
    logger.info(f"Mean Test AUPRC: {results_df['test_auprc'].mean():.4f} ± {results_df['test_auprc'].std():.4f}")


if __name__ == '__main__':
    main()