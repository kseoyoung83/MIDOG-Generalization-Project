"""
08_quick_test.py
빠른 디버깅: Macenko Fold1만 1 epoch 학습하여 전체 파이프라인 검증
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspace')
from myscripts import config

# ============= Dataset (간소화) =============
class MitosisDataset(Dataset):
    def __init__(self, patches_dir, fold_csv, transform=None, limit=None):
        self.patches_dir = Path(patches_dir)
        self.df = pd.read_csv(fold_csv)
        self.transform = transform
        
        self.samples = []
        for _, row in self.df.iterrows():
            scanner = row['scanner']
            file_num = row['file_number']
            
            pos_dir = self.patches_dir / scanner / 'positive'
            if pos_dir.exists():
                for patch_path in pos_dir.glob(f'{file_num:03d}_pos_*.png'):
                    self.samples.append((str(patch_path), 1))
            
            neg_dir = self.patches_dir / scanner / 'negative'
            if neg_dir.exists():
                for patch_path in neg_dir.glob(f'{file_num:03d}_neg_*.png'):
                    self.samples.append((str(patch_path), 0))
        
        # Limit samples for quick test
        if limit and len(self.samples) > limit:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), limit, replace=False)
            self.samples = [self.samples[i] for i in indices]
        
        print(f"Loaded {len(self.samples)} patches")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    return model

def quick_test():
    """Macenko Fold1, 1 epoch only"""
    
    print("="*60)
    print("QUICK TEST: Macenko Fold1, 1 epoch")
    print("="*60)
    
    # Paths
    method = 'macenko'
    fold = 1
    patches_dir = config.PROCESSED_DIR / f'patches_normalized/{method}'
    fold_train = config.METADATA_DIR / f'fold{fold}_train.csv'
    fold_val = config.METADATA_DIR / f'fold{fold}_val.csv'
    
    # Check
    if not patches_dir.exists():
        print(f"ERROR: {patches_dir} not found!")
        return
    
    print(f"\n1. Loading datasets (limited to 500 samples each)...")
    train_dataset = MitosisDataset(patches_dir, fold_train, 
                                   transform=get_transforms('train'), limit=500)
    val_dataset = MitosisDataset(patches_dir, fold_val, 
                                 transform=get_transforms('val'), limit=200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"\n2. Building model...")
    device = torch.device('cpu')
    model = get_model().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\n3. Training 1 epoch...")
    model.train()
    for images, labels in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"\n4. Validating...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    auroc = roc_auc_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Val AUROC: {auroc:.4f}")
    print(f"\nIf AUROC is reasonable (0.4-0.6), pipeline is working!")
    print(f"If AUROC is NaN or <0.3, check data loading.")

if __name__ == "__main__":
    quick_test()