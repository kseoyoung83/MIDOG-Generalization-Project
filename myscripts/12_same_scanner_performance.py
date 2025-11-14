#!/usr/bin/env python3
"""
12_same_scanner_performance.py
Same-scanner 성능 측정 (Non-LOSO)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
from tqdm import tqdm

sys.path.append('/workspace/myscripts')
from config import *

class PatchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['patch_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = row['label']
        return img, label

def create_same_scanner_splits(patches_list_path, output_dir):
    """각 scanner별로 80/10/10 split"""
    df = pd.read_csv(patches_list_path)
    
    splits = {}
    for scanner in ['Hamamatsu-XR', 'Hamamatsu-S360', 'Aperio-CS2']:
        df_s = df[df['scanner'] == scanner].copy()
        patients = df_s['case_id'].unique()
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(patients)
        
        n = len(patients)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_pts = patients[:n_train]
        val_pts = patients[n_train:n_train+n_val]
        test_pts = patients[n_train+n_val:]
        
        splits[scanner] = {
            'train': df_s[df_s['case_id'].isin(train_pts)],
            'val': df_s[df_s['case_id'].isin(val_pts)],
            'test': df_s[df_s['case_id'].isin(test_pts)]
        }
        
        for split_name, split_df in splits[scanner].items():
            split_df.to_csv(output_dir / f'{scanner}_{split_name}.csv', index=False)
        
        print(f"{scanner}: {len(train_pts)} train, {len(val_pts)} val, {len(test_pts)} test patients")
    
    return splits

def train_same_scanner_model(scanner, splits, output_dir, max_epochs=20):
    """1개 scanner로 학습"""
    device = torch.device('cpu')
    
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    model = model.to(device)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    train_dataset = PatchDataset(splits['train'], transform_train)
    val_dataset = PatchDataset(splits['val'], transform_test)
    test_dataset = PatchDataset(splits['test'], transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_val_auroc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_auroc = roc_auc_score(val_labels, val_probs)
        print(f"Epoch {epoch+1}: Val AUROC = {val_auroc:.4f}")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), output_dir / f'{scanner}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(output_dir / f'{scanner}_best.pth'))
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_auroc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)
    
    print(f"\n{scanner} - Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
    
    return {
        'scanner': scanner,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'best_val_auroc': best_val_auroc
    }

def main():
    print("="*60)
    print("SAME-SCANNER PERFORMANCE MEASUREMENT")
    print("="*60)
    
    output_dir = RESULTS_DIR / 'same_scanner'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patches_list = DATA_DIR / 'metadata' / 'patches_list.csv'
    
    if not patches_list.exists():
        print("ERROR: patches_list.csv not found!")
        print("Run: python3 << 'EOF'")
        print("import pandas as pd")
        print("from pathlib import Path")
        print("patches = []")
        print("for p in Path('data/processed/patches').rglob('*.png'):")
        print("    parts = p.stem.split('_')")
        print("    label = 1 if 'pos' in p.stem else 0")
        print("    scanner = p.parts[-3]")
        print("    case_id = parts[0]")
        print("    patches.append({'patch_path': str(p), 'case_id': case_id, 'scanner': scanner, 'label': label})")
        print("pd.DataFrame(patches).to_csv('data/metadata/patches_list.csv', index=False)")
        print("EOF")
        return
    
    print("\n1. Creating same-scanner splits...")
    splits_all = create_same_scanner_splits(patches_list, output_dir)
    
    print("\n2. Training same-scanner models...")
    results = []
    for scanner in ['Hamamatsu-XR', 'Hamamatsu-S360', 'Aperio-CS2']:
        print(f"\n{'='*60}")
        print(f"Training {scanner}...")
        print('='*60)
        result = train_same_scanner_model(scanner, splits_all[scanner], output_dir)
        results.append(result)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'same_scanner_results.csv', index=False)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    
    mean_same = df_results['test_auroc'].mean()
    std_same = df_results['test_auroc'].std()
    cross_auroc = 0.5068
    
    print(f"\nMean same-scanner AUROC: {mean_same:.4f} ± {std_same:.4f}")
    print(f"Cross-scanner AUROC: {cross_auroc:.4f}")
    print(f"Reduction: {((mean_same - cross_auroc) / mean_same * 100):.1f}%")

if __name__ == '__main__':
    main()