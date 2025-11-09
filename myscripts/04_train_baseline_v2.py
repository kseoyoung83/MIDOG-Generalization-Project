#!/usr/bin/env python3
"""
MIDOG 2021 Baseline Model Training V2
- Strong regularization
- Class weights
- Enhanced augmentation
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
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import logging

sys.path.append('/workspace/myscripts')
from config import *

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    np.random.seed(RANDOM_SEED + worker_id)
    torch.manual_seed(RANDOM_SEED + worker_id)


class MitosisDataset(Dataset):
    SCANNER_MAP = {
        'Aperio-CS2': 'Aperio-CS2',
        'Hamamatsu-XR': 'Hamamatsu-XR',
        'Hamamatsu-S360': 'Hamamatsu-S360'
    }
    
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.patches_dir = PROCESSED_DIR / 'patches'
        self.patch_list = []
        
        for _, row in self.df.iterrows():
            scanner_folder = self.SCANNER_MAP.get(row['scanner'], row['scanner'])
            file_num = row['file_number']
            
            for label_folder, label_value in [('positive', 1), ('negative', 0)]:
                patch_dir = self.patches_dir / scanner_folder / label_folder
                if not patch_dir.exists():
                    continue
                
                for patch_file in patch_dir.glob(f'{file_num:03d}_*.png'):
                    self.patch_list.append({
                        'path': patch_file,
                        'label': label_value
                    })
        
        logger.info(f"Found {len(self.patch_list)} patches from {len(self.df)} images")
        
        # Calculate class weights
        labels = [p['label'] for p in self.patch_list]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        self.pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Class distribution: pos={n_pos}, neg={n_neg}, pos_weight={self.pos_weight:.2f}")
        
    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, idx):
        patch_info = self.patch_list[idx]
        img = Image.open(patch_info['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, patch_info['label']


def build_model(dropout=0.5):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    
    # Add dropout
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 2)
    )
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.0
        
    return epoch_loss, auroc


def validate(model, loader, criterion, device):
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
    
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except:
        auroc, auprc = 0.0, 0.0
    
    return epoch_loss, auroc, auprc, all_labels, all_probs


def train_fold(fold_num, device):
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Fold {fold_num}")
    logger.info(f"{'='*50}")
    
    train_csv = METADATA_DIR / f'fold{fold_num}_train.csv'
    val_csv = METADATA_DIR / f'fold{fold_num}_val.csv'
    test_csv = METADATA_DIR / f'fold{fold_num}_test.csv'
    
    # Enhanced transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MitosisDataset(train_csv, train_transform)
    val_dataset = MitosisDataset(val_csv, val_transform)
    test_dataset = MitosisDataset(test_csv, val_transform)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    # Model with dropout
    model = build_model(dropout=0.5).to(device)
    
    # Weighted CrossEntropyLoss
    pos_weight = torch.tensor([train_dataset.pos_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, train_dataset.pos_weight]).to(device))
    
    # Lower learning rate, stronger weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    best_auroc = 0.0
    patience_counter = 0
    patience = 7  # Reduced patience
    
    history = {
        'train_loss': [], 'train_auroc': [],
        'val_loss': [], 'val_auroc': [], 'val_auprc': []
    }
    
    for epoch in range(30):  # Reduced epochs
        logger.info(f"\nEpoch {epoch+1}/30")
        
        train_loss, train_auroc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc, val_auprc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_auroc)
        
        history['train_loss'].append(train_loss)
        history['train_auroc'].append(train_auroc)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_auprc'].append(val_auprc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_counter = 0
            
            model_path = RESULTS_DIR / 'models' / f'fold{fold_num}_best_v2.pth'
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
        
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    
    # Load best model for test
    checkpoint = torch.load(RESULTS_DIR / 'models' / f'fold{fold_num}_best_v2.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("\nEvaluating on test set...")
    test_loss, test_auroc, test_auprc, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
    
    results = {
        'fold': fold_num,
        'best_val_auroc': best_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'epochs_trained': epoch + 1
    }
    
    return results, history, test_labels, test_probs


def main():
    logger.info("Starting MIDOG Baseline Training V2")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    device = torch.device(DEVICE)
    
    all_results = []
    
    for fold in [1, 2, 3]:
        results, history, test_labels, test_probs = train_fold(fold, device)
        all_results.append(results)
        
        history_df = pd.DataFrame(history)
        history_path = RESULTS_DIR / 'metrics' / f'fold{fold}_history_v2.csv'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_path, index=False)
        
        pred_df = pd.DataFrame({
            'label': test_labels,
            'prob': test_probs
        })
        pred_path = RESULTS_DIR / 'metrics' / f'fold{fold}_predictions_v2.csv'
        pred_df.to_csv(pred_path, index=False)
    
    results_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / 'metrics' / 'baseline_summary_v2.csv'
    results_df.to_csv(summary_path, index=False)
    
    logger.info("\n" + "="*50)
    logger.info("Baseline Training V2 Complete")
    logger.info("="*50)
    logger.info(f"\n{results_df}")
    logger.info(f"\nMean Test AUROC: {results_df['test_auroc'].mean():.4f} ± {results_df['test_auroc'].std():.4f}")
    logger.info(f"Mean Test AUPRC: {results_df['test_auprc'].mean():.4f} ± {results_df['test_auprc'].std():.4f}")


if __name__ == '__main__':
    main()