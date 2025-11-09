"""
03_generate_loso_folds.py
LOSO (Leave-One-Scanner-Out) Cross-Validation Fold 생성

목적:
- 3개의 Fold 생성 (Fold1, Fold2, Fold3)
- 각 Fold마다 1개 스캐너 → Test, 나머지 2개 스캐너 → Train/Val
- 데이터 누수 방지: 스캐너 완전 분리, 환자 ID 단위 분할
- 재현성 보장: seed=42 고정

출력:
- data/metadata/fold1_train.csv, fold1_val.csv, fold1_test.csv
- data/metadata/fold2_train.csv, fold2_val.csv, fold2_test.csv
- data/metadata/fold3_train.csv, fold3_val.csv, fold3_test.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

sys.path.append('/workspace')
from myscripts import config

# 재현성 보장
config.set_seed(config.RANDOM_SEED)


def validate_scanner_separation(train_df, val_df, test_df, fold_name):
    """
    스캐너 완전 분리 검증 - 데이터 누수 방지의 핵심
    
    CRITICAL: test 스캐너가 train/val에 절대 포함되지 않아야 함
    """
    train_scanners = set(train_df['scanner'].unique())
    val_scanners = set(val_df['scanner'].unique())
    test_scanners = set(test_df['scanner'].unique())
    
    # 검증 1: Test 스캐너가 Train/Val에 없어야 함
    assert len(test_scanners & train_scanners) == 0, \
        f"{fold_name}: Test scanner leaked into train set!"
    assert len(test_scanners & val_scanners) == 0, \
        f"{fold_name}: Test scanner leaked into val set!"
    
    # 검증 2: Train과 Val은 같은 스캐너 집합을 가져야 함
    assert train_scanners == val_scanners, \
        f"{fold_name}: Train and Val must share same scanners!"
    
    print(f"  ✓ Scanner separation validated")
    print(f"    - Train/Val scanners: {sorted(train_scanners)}")
    print(f"    - Test scanners: {sorted(test_scanners)}")


def validate_patient_separation(train_df, val_df, test_df, fold_name):
    """
    환자 ID 단위 분할 검증 - 동일 환자가 train/val/test에 중복되지 않아야 함
    
    NOTE: 현재 MIDOG 2021 데이터는 image_id가 patient_id 역할
          실제 연구에서는 patient_id 컬럼을 사용해야 함
    """
    train_patients = set(train_df['image_id'].unique())
    val_patients = set(val_df['image_id'].unique())
    test_patients = set(test_df['image_id'].unique())
    
    # 검증: 모든 집합이 서로소여야 함
    assert len(train_patients & val_patients) == 0, \
        f"{fold_name}: Patient overlap between train and val!"
    assert len(train_patients & test_patients) == 0, \
        f"{fold_name}: Patient overlap between train and test!"
    assert len(val_patients & test_patients) == 0, \
        f"{fold_name}: Patient overlap between val and test!"
    
    print(f"  ✓ Patient-level separation validated")
    print(f"    - Train: {len(train_patients)} patients")
    print(f"    - Val: {len(val_patients)} patients")
    print(f"    - Test: {len(test_patients)} patients")


def generate_single_fold(metadata_df, fold_config, output_dir):
    """
    단일 LOSO Fold 생성
    
    Args:
        metadata_df: 전체 메타데이터
        fold_config: LOSO_FOLDS의 단일 fold 설정
        output_dir: CSV 저장 경로
    
    Returns:
        train_df, val_df, test_df
    """
    fold_name = fold_config['name']
    test_scanner = fold_config['test_scanner']
    train_scanners = fold_config['train_scanners']
    
    print(f"\n{'='*60}")
    print(f"Generating {fold_name}")
    print(f"{'='*60}")
    print(f"Test scanner: {test_scanner}")
    print(f"Train scanners: {train_scanners}")
    
    # 1. Test set: test_scanner의 모든 이미지
    test_df = metadata_df[metadata_df['scanner'] == test_scanner].copy()
    
    # 2. Train+Val candidates: train_scanners의 모든 이미지
    trainval_df = metadata_df[
        metadata_df['scanner'].isin(train_scanners)
    ].copy()
    
    # 3. Train/Val 분할 (80/20) - 환자 ID 단위 분할
    # CRITICAL: 동일 환자가 train/val에 중복되지 않도록 stratify 적용
    unique_images = trainval_df['image_id'].unique()
    
    # 스캐너별 비율 유지하기 위한 stratify key 생성
    # NOTE: 현재는 스캐너만 고려하지만, mitosis 개수도 고려하면 더 균형잡힌 split 가능
    # 예: df['mitosis_category'] = pd.cut(df['num_mitoses'], bins=[0, 10, 30, 100])
    stratify_labels = []
    for img_id in unique_images:
        scanner = trainval_df[trainval_df['image_id'] == img_id]['scanner'].iloc[0]
        stratify_labels.append(scanner)
        # 개선 옵션: 
        # num_mitoses = trainval_df[trainval_df['image_id'] == img_id]['num_mitoses'].iloc[0]
        # mitosis_cat = 'low' if num_mitoses < 10 else 'medium' if num_mitoses < 30 else 'high'
        # stratify_labels.append(f"{scanner}_{mitosis_cat}")
    
    train_images, val_images = train_test_split(
        unique_images,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=stratify_labels
    )
    
    train_df = trainval_df[trainval_df['image_id'].isin(train_images)].copy()
    val_df = trainval_df[trainval_df['image_id'].isin(val_images)].copy()
    
    # 4. 검증: 데이터 누수 방지
    validate_scanner_separation(train_df, val_df, test_df, fold_name)
    validate_patient_separation(train_df, val_df, test_df, fold_name)
    
    # 5. 통계 출력
    print(f"\n  Dataset Statistics:")
    print(f"  {'Split':<10} {'Images':<10} {'Mitoses':<10} {'Pos Patches (est.)':<20}")
    print(f"  {'-'*60}")
    
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_images = len(split_df)
        n_mitoses = split_df['num_mitoses'].sum()
        est_pos_patches = n_mitoses  # 1 mitosis ≈ 1 positive patch
        
        print(f"  {split_name:<10} {n_images:<10} {n_mitoses:<10} {est_pos_patches:<20}")
    
    # 6. 스캐너별 분포
    print(f"\n  Scanner Distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        scanner_counts = split_df['scanner'].value_counts().to_dict()
        print(f"    {split_name}: {scanner_counts}")
    
    # 7. CSV 저장
    # 파일명 생성: "Fold1_TestA" → "fold1"
    fold_number = fold_name.split('_')[0].lower()  # "fold1", "fold2", "fold3"
    
    train_csv = output_dir / f"{fold_number}_train.csv"
    val_csv = output_dir / f"{fold_number}_val.csv"
    test_csv = output_dir / f"{fold_number}_test.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n  ✓ Saved:")
    print(f"    - {train_csv}")
    print(f"    - {val_csv}")
    print(f"    - {test_csv}")
    
    return train_df, val_df, test_df


def generate_all_folds(metadata_csv, output_dir):
    """
    3개의 LOSO Fold 생성
    
    MIDOG 2021: 3개 스캐너 (Leica 제외)
    - Fold1: Test=Hamamatsu-XR, Train/Val=Hamamatsu-S360+Aperio-CS2
    - Fold2: Test=Hamamatsu-S360, Train/Val=Hamamatsu-XR+Aperio-CS2
    - Fold3: Test=Aperio-CS2, Train/Val=Hamamatsu-XR+Hamamatsu-S360
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 메타데이터 로드
    print(f"Loading metadata from {metadata_csv}...")
    metadata_df = pd.read_csv(metadata_csv)
    
    # Leica-GT450 제외 (unlabeled)
    metadata_df = metadata_df[metadata_df['scanner'] != 'Leica-GT450'].copy()
    
    print(f"\n{'='*60}")
    print(f"MIDOG 2021 LOSO Fold Generation")
    print(f"{'='*60}")
    print(f"Total images (excluding Leica): {len(metadata_df)}")
    print(f"Total mitoses: {metadata_df['num_mitoses'].sum()}")
    print(f"\nImages per scanner:")
    print(metadata_df['scanner'].value_counts())
    
    # 3개의 Fold 생성
    all_results = {}
    
    for fold_config in config.LOSO_FOLDS:
        train_df, val_df, test_df = generate_single_fold(
            metadata_df, 
            fold_config, 
            output_dir
        )
        
        all_results[fold_config['name']] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    # 전체 요약
    print(f"\n{'='*60}")
    print(f"LOSO Fold Generation Complete")
    print(f"{'='*60}")
    
    summary_data = []
    for fold_name, splits in all_results.items():
        for split_name, split_df in splits.items():
            summary_data.append({
                'Fold': fold_name,
                'Split': split_name,
                'Images': len(split_df),
                'Mitoses': split_df['num_mitoses'].sum(),
                'Scanners': ', '.join(sorted(split_df['scanner'].unique()))
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'loso_folds_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\n✓ Summary saved to: {summary_csv}")
    
    # CRITICAL 검증: 모든 Fold에서 Test 스캐너가 고유해야 함
    test_scanners = []
    for fold_name, splits in all_results.items():
        test_scanner = splits['test']['scanner'].unique()[0]
        test_scanners.append(test_scanner)
    
    assert len(test_scanners) == len(set(test_scanners)), \
        "ERROR: Test scanners are not unique across folds!"
    
    print(f"\n✓ VALIDATION PASSED: All folds have unique test scanners")
    print(f"  Test scanners by fold: {test_scanners}")
    
    return all_results


def main():
    """메인 실행"""
    metadata_csv = config.METADATA_DIR / "metadata.csv"
    output_dir = config.METADATA_DIR
    
    if not metadata_csv.exists():
        print(f"ERROR: {metadata_csv} not found!")
        print(f"Please run 01_parse_metadata.py first.")
        return
    
    print(f"\n{'#'*60}")
    print(f"# LOSO Fold Generation Script")
    print(f"# Random Seed: {config.RANDOM_SEED}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}\n")
    
    results = generate_all_folds(metadata_csv, output_dir)
    
    print(f"\n{'#'*60}")
    print(f"# LOSO Fold Generation Complete")
    print(f"# Files created:")
    print(f"#   - fold1_train.csv, fold1_val.csv, fold1_test.csv")
    print(f"#   - fold2_train.csv, fold2_val.csv, fold2_test.csv")
    print(f"#   - fold3_train.csv, fold3_val.csv, fold3_test.csv")
    print(f"#   - loso_folds_summary.csv")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()