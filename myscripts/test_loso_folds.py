"""
test_loso_folds.py
LOSO Fold 생성 결과 검증 스크립트

목적:
- 생성된 fold CSV 파일들이 올바른지 검증
- 데이터 누수가 없는지 확인
- 통계 검증

실행 방법:
    python myscripts/test_loso_folds.py
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append('/workspace')
from myscripts import config


def test_file_existence():
    """생성된 CSV 파일들이 존재하는지 확인"""
    print("\n" + "="*60)
    print("TEST 1: File Existence Check")
    print("="*60)
    
    required_files = [
        'fold1_train.csv', 'fold1_val.csv', 'fold1_test.csv',
        'fold2_train.csv', 'fold2_val.csv', 'fold2_test.csv',
        'fold3_train.csv', 'fold3_val.csv', 'fold3_test.csv',
        'loso_folds_summary.csv'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = config.METADATA_DIR / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n✓ TEST PASSED: All files exist")
    else:
        print("\n✗ TEST FAILED: Some files are missing")
    
    return all_exist


def test_scanner_separation():
    """스캐너 완전 분리 검증"""
    print("\n" + "="*60)
    print("TEST 2: Scanner Separation Validation")
    print("="*60)
    
    folds = [
        ('fold1', 'Hamamatsu-XR'),
        ('fold2', 'Hamamatsu-S360'),
        ('fold3', 'Aperio-CS2')
    ]
    
    all_passed = True
    
    for fold_prefix, expected_test_scanner in folds:
        train_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_train.csv")
        val_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_val.csv")
        test_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_test.csv")
        
        train_scanners = set(train_df['scanner'].unique())
        val_scanners = set(val_df['scanner'].unique())
        test_scanners = set(test_df['scanner'].unique())
        
        print(f"\n  {fold_prefix.upper()}:")
        print(f"    Train scanners: {sorted(train_scanners)}")
        print(f"    Val scanners: {sorted(val_scanners)}")
        print(f"    Test scanners: {sorted(test_scanners)}")
        
        # 검증 1: Test 스캐너가 Train/Val에 없어야 함
        train_leak = len(test_scanners & train_scanners) == 0
        val_leak = len(test_scanners & val_scanners) == 0
        
        # 검증 2: Test 스캐너가 예상과 일치해야 함
        correct_test = (len(test_scanners) == 1 and 
                       expected_test_scanner in test_scanners)
        
        # 검증 3: Train과 Val이 같은 스캐너 집합을 가져야 함
        same_trainval = train_scanners == val_scanners
        
        fold_passed = train_leak and val_leak and correct_test and same_trainval
        all_passed = all_passed and fold_passed
        
        status = "✓" if fold_passed else "✗"
        print(f"    {status} Scanner separation validated")
        
        if not train_leak:
            print(f"      ERROR: Test scanner leaked into train!")
        if not val_leak:
            print(f"      ERROR: Test scanner leaked into val!")
        if not correct_test:
            print(f"      ERROR: Expected test scanner {expected_test_scanner}, got {test_scanners}")
        if not same_trainval:
            print(f"      ERROR: Train and Val have different scanners!")
    
    if all_passed:
        print("\n✓ TEST PASSED: Scanner separation is correct")
    else:
        print("\n✗ TEST FAILED: Scanner separation has issues")
    
    return all_passed


def test_patient_separation():
    """환자 ID 단위 분할 검증"""
    print("\n" + "="*60)
    print("TEST 3: Patient-Level Separation Validation")
    print("="*60)
    
    fold_prefixes = ['fold1', 'fold2', 'fold3']
    all_passed = True
    
    for fold_prefix in fold_prefixes:
        train_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_train.csv")
        val_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_val.csv")
        test_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_test.csv")
        
        train_patients = set(train_df['image_id'].unique())
        val_patients = set(val_df['image_id'].unique())
        test_patients = set(test_df['image_id'].unique())
        
        print(f"\n  {fold_prefix.upper()}:")
        print(f"    Train patients: {len(train_patients)}")
        print(f"    Val patients: {len(val_patients)}")
        print(f"    Test patients: {len(test_patients)}")
        
        # 검증: 모든 집합이 서로소여야 함
        train_val_overlap = len(train_patients & val_patients)
        train_test_overlap = len(train_patients & test_patients)
        val_test_overlap = len(val_patients & test_patients)
        
        fold_passed = (train_val_overlap == 0 and 
                      train_test_overlap == 0 and 
                      val_test_overlap == 0)
        all_passed = all_passed and fold_passed
        
        status = "✓" if fold_passed else "✗"
        print(f"    {status} Patient separation validated")
        
        if train_val_overlap > 0:
            print(f"      ERROR: {train_val_overlap} patients overlap between train and val!")
        if train_test_overlap > 0:
            print(f"      ERROR: {train_test_overlap} patients overlap between train and test!")
        if val_test_overlap > 0:
            print(f"      ERROR: {val_test_overlap} patients overlap between val and test!")
    
    if all_passed:
        print("\n✓ TEST PASSED: Patient separation is correct")
    else:
        print("\n✗ TEST FAILED: Patient separation has issues")
    
    return all_passed


def test_data_coverage():
    """전체 데이터 커버리지 검증"""
    print("\n" + "="*60)
    print("TEST 4: Data Coverage Validation")
    print("="*60)
    
    # 원본 메타데이터
    metadata_df = pd.read_csv(config.METADATA_DIR / "metadata.csv")
    metadata_df = metadata_df[metadata_df['scanner'] != 'Leica-GT450']
    
    original_images = set(metadata_df['image_id'].unique())
    original_count = len(original_images)
    
    print(f"\n  Original dataset (excluding Leica):")
    print(f"    Total images: {original_count}")
    
    fold_prefixes = ['fold1', 'fold2', 'fold3']
    all_passed = True
    
    for fold_prefix in fold_prefixes:
        train_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_train.csv")
        val_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_val.csv")
        test_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_test.csv")
        
        fold_images = set(train_df['image_id'].unique()) | \
                     set(val_df['image_id'].unique()) | \
                     set(test_df['image_id'].unique())
        
        fold_count = len(fold_images)
        coverage = fold_count == original_count
        all_passed = all_passed and coverage
        
        status = "✓" if coverage else "✗"
        print(f"\n  {fold_prefix.upper()}:")
        print(f"    {status} Coverage: {fold_count}/{original_count} images")
        
        if not coverage:
            missing = original_images - fold_images
            extra = fold_images - original_images
            if missing:
                print(f"      ERROR: Missing {len(missing)} images: {sorted(list(missing))[:5]}...")
            if extra:
                print(f"      ERROR: Extra {len(extra)} images: {sorted(list(extra))[:5]}...")
    
    if all_passed:
        print("\n✓ TEST PASSED: All folds cover the entire dataset")
    else:
        print("\n✗ TEST FAILED: Some folds have incomplete coverage")
    
    return all_passed


def test_statistics():
    """통계 검증"""
    print("\n" + "="*60)
    print("TEST 5: Statistics Validation")
    print("="*60)
    
    fold_prefixes = ['fold1', 'fold2', 'fold3']
    
    for fold_prefix in fold_prefixes:
        train_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_train.csv")
        val_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_val.csv")
        test_df = pd.read_csv(config.METADATA_DIR / f"{fold_prefix}_test.csv")
        
        print(f"\n  {fold_prefix.upper()} Statistics:")
        print(f"    {'Split':<10} {'Images':<10} {'Mitoses':<10} {'Avg Mitoses/Image':<20}")
        print(f"    {'-'*60}")
        
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            n_images = len(split_df)
            n_mitoses = split_df['num_mitoses'].sum()
            avg_mitoses = n_mitoses / n_images if n_images > 0 else 0
            
            print(f"    {split_name:<10} {n_images:<10} {n_mitoses:<10} {avg_mitoses:<20.2f}")
    
    print("\n✓ Statistics displayed")
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "#"*60)
    print("# LOSO Fold Validation Tests")
    print("# Random Seed: {}".format(config.RANDOM_SEED))
    print("#"*60)
    
    tests = [
        test_file_existence,
        test_scanner_separation,
        test_patient_separation,
        test_data_coverage,
        test_statistics
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ ERROR in {test_func.__name__}: {str(e)}")
            results.append((test_func.__name__, False))
    
    # 최종 요약
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nLOSO folds are correctly generated and validated!")
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the errors above and regenerate folds.")
    
    return all_passed


def main():
    """메인 실행"""
    if not (config.METADATA_DIR / "metadata.csv").exists():
        print("ERROR: metadata.csv not found!")
        print("Please run 01_parse_metadata.py and 03_generate_loso_folds.py first.")
        return
    
    run_all_tests()


if __name__ == "__main__":
    main()