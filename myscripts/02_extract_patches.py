"""
02_extract_patches.py
PIL 직접 사용한 패치 추출
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append('/workspace')
from myscripts import config

def extract_patches(metadata_csv, output_dir):
    """PIL로 직접 패치 추출"""
    
    df = pd.read_csv(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = df[df['scanner'] != 'Leica-GT450']
    
    stats = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        image_path = config.RAW_DATA_DIR / row['filename']
        
        if not image_path.exists():
            continue
        
        scanner = row['scanner']
        file_num = row['file_number']
        
        pos_dir = output_dir / scanner / 'positive'
        neg_dir = output_dir / scanner / 'negative'
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)
        
        # TIFF 열기
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            
            # Positive 패치
            pos_count = 0
            if row['num_mitoses'] > 0:
                coords = eval(row['mitosis_coords'])
                
                for ann_idx, bbox in enumerate(coords):
                    x, y, w, h = bbox
                    center_x = int(x + w/2)
                    center_y = int(y + h/2)
                    
                    left = max(0, center_x - 112)
                    top = max(0, center_y - 112)
                    right = min(img_width, center_x + 112)
                    bottom = min(img_height, center_y + 112)
                    
                    if right - left < 200 or bottom - top < 200:
                        continue
                    
                    try:
                        patch = img.crop((left, top, right, bottom))
                        patch = patch.resize((224, 224))
                        
                        patch_path = pos_dir / f"{file_num:03d}_pos_{ann_idx:03d}.png"
                        patch.save(patch_path)
                        pos_count += 1
                    except:
                        continue
            
            # Negative 패치
            neg_target = max(pos_count * 3, 10)
            neg_count = 0
            attempts = 0
            
            np.random.seed(config.RANDOM_SEED + file_num)
            
            while neg_count < neg_target and attempts < neg_target * 10:
                attempts += 1
                
                x = np.random.randint(112, img_width - 112)
                y = np.random.randint(112, img_height - 112)
                
                try:
                    patch = img.crop((x - 112, y - 112, x + 112, y + 112))
                    patch = patch.resize((224, 224))
                    
                    # 배경 체크
                    patch_array = np.array(patch)
                    if np.mean(patch_array) > 220 or np.std(patch_array) < 10:
                        continue
                    
                    patch_path = neg_dir / f"{file_num:03d}_neg_{neg_count:03d}.png"
                    patch.save(patch_path)
                    neg_count += 1
                except:
                    continue
        
        stats.append({
            'filename': row['filename'],
            'scanner': scanner,
            'positive': pos_count,
            'negative': neg_count
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / 'extraction_stats.csv', index=False)
    
    print(f"\n=== Summary ===")
    print(f"Positive: {stats_df['positive'].sum()}")
    print(f"Negative: {stats_df['negative'].sum()}")
    for scanner in stats_df['scanner'].unique():
        sdf = stats_df[stats_df['scanner']==scanner]
        print(f"{scanner}: Pos={sdf['positive'].sum()}, Neg={sdf['negative'].sum()}")

def main():
    metadata_csv = config.METADATA_DIR / "metadata.csv"
    output_dir = config.PROCESSED_DIR / "patches"
    
    extract_patches(metadata_csv, output_dir)
    print(f"\n✓ Done: {output_dir}")

if __name__ == "__main__":
    main()