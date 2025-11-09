"""
06_select_target_image.py
Target image 선정 (정량적 기준)

기준:
1. H&E stain intensity median에 가장 가까운 이미지
2. Tissue ratio > 0.7
3. 각 스캐너별 대표 이미지 선정

출력: target_image_info.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

sys.path.append('/workspace')
from myscripts import config

def calculate_tissue_mask(img_array, threshold=220):
    """조직 영역 마스크 생성"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < threshold
    return tissue_mask

def calculate_he_intensity(img_array, tissue_mask):
    """
    H&E 채널별 intensity 계산
    
    Returns:
        dict: {'h_mean', 'e_mean', 'h_std', 'e_std'}
    """
    # RGB to OD (Optical Density)
    img_float = img_array.astype(np.float32) + 1
    od = -np.log(img_float / 256.0)
    
    # H&E 분리 (간단한 선형 변환)
    # Hematoxylin: 청색 (high B, low R)
    # Eosin: 분홍색 (high R, low B)
    
    h_channel = od[:, :, 2] - od[:, :, 0]  # B - R
    e_channel = od[:, :, 0] - od[:, :, 2]  # R - B
    
    # 조직 영역만 계산
    h_tissue = h_channel[tissue_mask]
    e_tissue = e_channel[tissue_mask]
    
    return {
        'h_mean': float(np.mean(h_tissue)),
        'h_std': float(np.std(h_tissue)),
        'e_mean': float(np.mean(e_tissue)),
        'e_std': float(np.std(e_tissue))
    }

def process_images():
    """모든 이미지 처리"""
    metadata_path = config.METADATA_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)
    
    # Leica 제외
    df = df[df['scanner'] != 'Leica-GT450']
    
    results = []
    
    print("Processing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = config.RAW_DATA_DIR / row['filename']
        
        if not img_path.exists():
            continue
        
        # 이미지 로드 (downsampled)
        with Image.open(img_path) as img:
            # 1/4 해상도로 빠르게 계산
            img_small = img.resize((img.width // 4, img.height // 4))
            img_array = np.array(img_small)
        
        # 조직 마스크
        tissue_mask = calculate_tissue_mask(img_array)
        tissue_ratio = tissue_mask.sum() / tissue_mask.size
        
        # 최소 조직 비율 체크
        if tissue_ratio < 0.3:
            continue
        
        # H&E intensity
        he_stats = calculate_he_intensity(img_array, tissue_mask)
        
        results.append({
            'filename': row['filename'],
            'file_number': row['file_number'],
            'scanner': row['scanner'],
            'tissue_ratio': float(tissue_ratio),
            'h_mean': he_stats['h_mean'],
            'h_std': he_stats['h_std'],
            'e_mean': he_stats['e_mean'],
            'e_std': he_stats['e_std']
        })
    
    return pd.DataFrame(results)

def select_target_per_scanner(df):
    """각 스캐너별 target image 선정"""
    targets = {}
    
    for scanner in df['scanner'].unique():
        scanner_df = df[df['scanner'] == scanner]
        
        # 조직 비율 필터링
        scanner_df = scanner_df[scanner_df['tissue_ratio'] > 0.7]
        
        if len(scanner_df) == 0:
            print(f"WARNING: {scanner} - 조직 비율 > 0.7인 이미지 없음. 기준 완화")
            scanner_df = df[df['scanner'] == scanner]
            scanner_df = scanner_df[scanner_df['tissue_ratio'] > 0.5]
        
        # H&E median 계산
        h_median = scanner_df['h_mean'].median()
        e_median = scanner_df['e_mean'].median()
        
        # Median에 가장 가까운 이미지 찾기
        scanner_df['distance'] = np.sqrt(
            (scanner_df['h_mean'] - h_median)**2 + 
            (scanner_df['e_mean'] - e_median)**2
        )
        
        target = scanner_df.nsmallest(1, 'distance').iloc[0]
        
        targets[scanner] = {
            'filename': target['filename'],
            'file_number': int(target['file_number']),
            'tissue_ratio': float(target['tissue_ratio']),
            'h_mean': float(target['h_mean']),
            'h_std': float(target['h_std']),
            'e_mean': float(target['e_mean']),
            'e_std': float(target['e_std']),
            'distance_to_median': float(target['distance']),
            'scanner_h_median': float(h_median),
            'scanner_e_median': float(e_median),
            'n_candidates': len(scanner_df)
        }
        
        print(f"\n{scanner}:")
        print(f"  Target: {target['filename']}")
        print(f"  Tissue ratio: {target['tissue_ratio']:.3f}")
        print(f"  H mean: {target['h_mean']:.4f} (median: {h_median:.4f})")
        print(f"  E mean: {target['e_mean']:.4f} (median: {e_median:.4f})")
        print(f"  Distance: {target['distance']:.6f}")
    
    return targets

def main():
    print("="*80)
    print("06_select_target_image.py")
    print("Target image 선정 (정량적 기준)")
    print("="*80)
    
    # 이미지 처리
    print("\n[Step 1] Processing images...")
    df = process_images()
    print(f"✓ Processed {len(df)} images")
    
    # 통계 저장
    stats_path = config.RESULTS_DIR / "stain_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stats_path, index=False)
    print(f"✓ Saved: {stats_path}")
    
    # Target 선정
    print("\n[Step 2] Selecting target images...")
    targets = select_target_per_scanner(df)
    
    # 저장
    output_path = config.RESULTS_DIR / "target_image_info.json"
    with open(output_path, 'w') as f:
        json.dump(targets, f, indent=2)
    
    print(f"\n✓ Saved: {output_path}")
    
    # 요약
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for scanner, info in targets.items():
        print(f"{scanner}: {info['filename']}")
    
    print("\n다음 단계: 07_apply_stain_normalization.py")
    print("="*80)

if __name__ == "__main__":
    main()