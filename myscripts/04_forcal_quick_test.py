#!/usr/bin/env python3
"""
04_focal_quick_test.py
Focal Loss 효과 빠른 검증 - Fold1만, 5 epoch
소요 시간: ~30분
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append('/workspace/myscripts')
from config import *


# ============= Focal Loss =============
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============= Dataset =============
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
        
        labels = [p['label'] for p in self.patch_list]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        self.pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"Loaded {len(self.patch_list)} patches (pos={n_pos}, neg={n_neg}, weight={self.pos_weight:.2f})")
    
    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, idx):
        patch_info = self.patch_list[idx]
        img = Image.open(patch_info['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, patch_info['label']


# ============= Model =============
def build_model(dropout=0.5):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 2)
    )
    return model


# ============= Training =============
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
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
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
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return epoch_loss, auroc


# ============= Main =============
def main():
    print("="*60)
    print("FOCAL LOSS QUICK TEST")
    print("Fold1, 5 epochs, ~30분 소요")
    print("="*60)
    
    fold = 1
    train_csv = METADATA_DIR / f'fold{fold}_train.csv'
    val_csv = METADATA_DIR / f'fold{fold}_val.csv'
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MitosisDataset(train_csv, train_transform)
    val_dataset = MitosisDataset(val_csv, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    device = torch.device('cpu')
    model = build_model(dropout=0.5).to(device)
    
    # Focal Loss with class weights
    class_weights = torch.tensor([1.0, train_dataset.pos_weight]).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("\n" + "="*60)
    print("Training with Focal Loss...")
    print("="*60)
    
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        train_loss, train_auroc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}")
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)
    print(f"Final Val AUROC: {val_auroc:.4f}")
    print("\n평가:")
    if val_auroc > 0.55:
        print("✓ 성공! AUROC > 0.55 → Focal Loss 효과 확인")
        print("  → 04_train_baseline_v3_focal.py로 전체 재학습 권장")
    elif val_auroc > 0.52:
        print("△ 개선됨. AUROC > 0.52 → 약간의 효과")
        print("  → 전체 학습 고려 가능")
    else:
        print("✗ 효과 없음. AUROC ≈ 0.50")
        print("  → Focal Loss 외 다른 방법 필요 (패치 크기 증가 등)")


if __name__ == '__main__':
    main()