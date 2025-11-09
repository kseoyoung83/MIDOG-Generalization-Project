"""
05_h1_statistical_test_both.py
H1 가설 검증: Baseline V1과 V2 둘 다 분석

출력:
- V1 결과
- V2 결과
- V1 vs V2 비교
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspace')
from myscripts import config

def bootstrap_auroc(y_true, y_pred_proba, n_iterations=1000, random_state=42):
    """Bootstrap resampling으로 AUROC 신뢰구간 추정"""
    np.random.seed(random_state)
    bootstrap_aurocs = []
    
    n_samples = len(y_true)
    
    for i in range(n_iterations):
        indices = resample(range(n_samples), n_samples=n_samples, 
                          random_state=random_state + i)
        
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred_proba.iloc[indices] if hasattr(y_pred_proba, 'iloc') else y_pred_proba[indices]
        
        try:
            auroc = roc_auc_score(y_true_boot, y_pred_boot)
            bootstrap_aurocs.append(auroc)
        except:
            continue
    
    bootstrap_aurocs = np.array(bootstrap_aurocs)
    
    return {
        'auroc_mean': np.mean(bootstrap_aurocs),
        'auroc_std': np.std(bootstrap_aurocs),
        'auroc_ci_lower': np.percentile(bootstrap_aurocs, 2.5),
        'auroc_ci_upper': np.percentile(bootstrap_aurocs, 97.5),
        'bootstrap_aurocs': bootstrap_aurocs
    }

def load_baseline_results_with_bootstrap(version='v1', n_iterations=1000):
    """
    Baseline 결과 로드 + Bootstrap
    
    Args:
        version: 'v1' or 'v2'
        n_iterations: bootstrap 반복 횟수
    """
    results = {}
    suffix = '' if version == 'v1' else '_v2'
    
    for fold_idx in [1, 2, 3]:
        pred_path = config.METRICS_DIR / f"fold{fold_idx}_predictions{suffix}.csv"
        
        if not pred_path.exists():
            print(f"  WARNING: {pred_path} not found. Skipping fold{fold_idx}.")
            continue
        
        print(f"  Processing fold{fold_idx}...")
        df = pd.read_csv(pred_path)
        
        # Test scanner 결정
        fold_info = config.LOSO_FOLDS[fold_idx - 1]
        test_scanner = fold_info['test_scanner']
        
        # Point estimate AUROC
        auroc_point = roc_auc_score(df['label'], df['prob'])
        
        # Bootstrap AUROC
        boot_result = bootstrap_auroc(
            df['label'], 
            df['prob'],
            n_iterations=n_iterations,
            random_state=config.RANDOM_SEED
        )
        
        results[f'fold{fold_idx}'] = {
            'test_scanner': test_scanner,
            'auroc_point': auroc_point,
            'auroc_mean': boot_result['auroc_mean'],
            'auroc_std': boot_result['auroc_std'],
            'auroc_ci_lower': boot_result['auroc_ci_lower'],
            'auroc_ci_upper': boot_result['auroc_ci_upper'],
            'bootstrap_aurocs': boot_result['bootstrap_aurocs']
        }
        
        print(f"    {test_scanner}: {auroc_point:.4f} "
              f"[{boot_result['auroc_ci_lower']:.4f}, {boot_result['auroc_ci_upper']:.4f}]")
    
    return results

def prepare_bootstrap_data_for_anova(results):
    """Bootstrap AUROC을 ANOVA용 데이터로 변환"""
    records = []
    
    for fold_name, fold_data in results.items():
        scanner = fold_data['test_scanner']
        bootstrap_aurocs = fold_data['bootstrap_aurocs']
        
        for i, auroc in enumerate(bootstrap_aurocs):
            records.append({
                'fold': fold_name,
                'scanner': scanner,
                'auroc': auroc,
                'bootstrap_iter': i
            })
    
    return pd.DataFrame(records)

def one_way_anova_bootstrap(df, alpha=0.05):
    """Bootstrap AUROC으로 ANOVA"""
    groups = [group['auroc'].values for name, group in df.groupby('scanner')]
    f_stat, p_value = stats.f_oneway(*groups)
    
    reject_h0 = p_value < alpha
    
    # 효과 크기
    grand_mean = df['auroc'].mean()
    ssb = sum([len(group) * (group['auroc'].mean() - grand_mean)**2 
               for name, group in df.groupby('scanner')])
    sst = sum((df['auroc'] - grand_mean)**2)
    eta_squared = ssb / sst if sst > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'alpha': alpha,
        'reject_h0': reject_h0,
        'eta_squared': eta_squared
    }

def tukey_hsd_posthoc_bootstrap(df, alpha=0.05):
    """Tukey HSD 사후 검정"""
    tukey = pairwise_tukeyhsd(df['auroc'], df['scanner'], alpha=alpha)
    return pd.DataFrame(data=tukey.summary().data[1:], 
                       columns=tukey.summary().data[0])

def print_comparison(results_v1, results_v2, anova_v1, anova_v2):
    """V1 vs V2 비교"""
    print("\n" + "="*80)
    print("V1 vs V2 비교")
    print("="*80)
    
    print("\n[Point Estimate AUROC]")
    print("-"*80)
    print(f"{'Fold':<10} {'Scanner':<20} {'V1':<10} {'V2':<10} {'Diff':<10}")
    print("-"*80)
    
    v1_aurocs = []
    v2_aurocs = []
    
    for fold_name in ['fold1', 'fold2', 'fold3']:
        v1_data = results_v1[fold_name]
        v2_data = results_v2[fold_name]
        
        v1_auroc = v1_data['auroc_point']
        v2_auroc = v2_data['auroc_point']
        diff = v2_auroc - v1_auroc
        
        v1_aurocs.append(v1_auroc)
        v2_aurocs.append(v2_auroc)
        
        print(f"{fold_name:<10} {v1_data['test_scanner']:<20} "
              f"{v1_auroc:<10.4f} {v2_auroc:<10.4f} {diff:+.4f}")
    
    print("-"*80)
    print(f"{'Mean':<10} {'':<20} "
          f"{np.mean(v1_aurocs):<10.4f} {np.mean(v2_aurocs):<10.4f} "
          f"{np.mean(v2_aurocs) - np.mean(v1_aurocs):+.4f}")
    
    print("\n[ANOVA 결과]")
    print("-"*80)
    print(f"{'Version':<10} {'F-stat':<12} {'p-value':<12} {'η²':<10} {'H0 기각':<10}")
    print("-"*80)
    print(f"{'V1':<10} {anova_v1['f_statistic']:<12.4f} "
          f"{anova_v1['p_value']:<12.6f} {anova_v1['eta_squared']:<10.4f} "
          f"{str(anova_v1['reject_h0']):<10}")
    print(f"{'V2':<10} {anova_v2['f_statistic']:<12.4f} "
          f"{anova_v2['p_value']:<12.6f} {anova_v2['eta_squared']:<10.4f} "
          f"{str(anova_v2['reject_h0']):<10}")
    
    print("\n[해석]")
    print("-"*80)
    
    mean_diff = np.mean(v2_aurocs) - np.mean(v1_aurocs)
    
    print(f"1. 성능 변화: V2가 V1 대비 {mean_diff:+.4f} {'향상' if mean_diff > 0 else '하락'}")
    if abs(mean_diff) < 0.01:
        print("   → 실질적으로 동일한 성능 (차이 < 0.01)")
    
    print(f"\n2. Domain Shift 입증:")
    if anova_v1['reject_h0'] or anova_v2['reject_h0']:
        print("   ✓ 최소 하나의 버전에서 H0 기각")
        if anova_v1['reject_h0'] and anova_v2['reject_h0']:
            print("   ✓ V1, V2 모두 Domain shift 입증")
        elif anova_v1['reject_h0']:
            print("   ✓ V1만 Domain shift 입증")
        else:
            print("   ✓ V2만 Domain shift 입증")
    else:
        print("   ✗ V1, V2 모두 Domain shift 입증 실패")
        print("   → 원인: 모든 스캐너에서 동일하게 낮은 성능")
    
    print(f"\n3. 권장 사항:")
    if anova_v2['reject_h0']:
        print("   → V2 결과로 연구 진행")
        print("   → 다음: 06_select_target_image.py")
    else:
        print("   → Domain shift 입증 실패")
        print("   → 전략: Stain norm으로 성능 향상 후 재검증")
        print("   → 다음: 06_select_target_image.py (Macenko 테스트)")

def save_results_both(results_v1, results_v2, anova_v1, anova_v2, output_dir):
    """결과 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # V1 Bootstrap AUROC
    boot_records_v1 = []
    for fold_name, fold_data in results_v1.items():
        for i, auroc in enumerate(fold_data['bootstrap_aurocs']):
            boot_records_v1.append({
                'version': 'v1',
                'fold': fold_name,
                'scanner': fold_data['test_scanner'],
                'bootstrap_iter': i,
                'auroc': auroc
            })
    
    # V2 Bootstrap AUROC
    boot_records_v2 = []
    for fold_name, fold_data in results_v2.items():
        for i, auroc in enumerate(fold_data['bootstrap_aurocs']):
            boot_records_v2.append({
                'version': 'v2',
                'fold': fold_name,
                'scanner': fold_data['test_scanner'],
                'bootstrap_iter': i,
                'auroc': auroc
            })
    
    # 통합
    boot_df = pd.DataFrame(boot_records_v1 + boot_records_v2)
    boot_path = output_dir / "h1_bootstrap_aurocs_both.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"✓ Saved: {boot_path}")
    
    # ANOVA 결과
    anova_df = pd.DataFrame([
        {
            'version': 'v1',
            'f_statistic': anova_v1['f_statistic'],
            'p_value': anova_v1['p_value'],
            'eta_squared': anova_v1['eta_squared'],
            'reject_h0': anova_v1['reject_h0']
        },
        {
            'version': 'v2',
            'f_statistic': anova_v2['f_statistic'],
            'p_value': anova_v2['p_value'],
            'eta_squared': anova_v2['eta_squared'],
            'reject_h0': anova_v2['reject_h0']
        }
    ])
    anova_path = output_dir / "h1_anova_results_both.csv"
    anova_df.to_csv(anova_path, index=False)
    print(f"✓ Saved: {anova_path}")
    
    # 비교 요약
    comparison_records = []
    for fold_name in ['fold1', 'fold2', 'fold3']:
        v1_data = results_v1[fold_name]
        v2_data = results_v2[fold_name]
        comparison_records.append({
            'fold': fold_name,
            'scanner': v1_data['test_scanner'],
            'auroc_v1': v1_data['auroc_point'],
            'auroc_v2': v2_data['auroc_point'],
            'diff': v2_data['auroc_point'] - v1_data['auroc_point'],
            'ci_v1_lower': v1_data['auroc_ci_lower'],
            'ci_v1_upper': v1_data['auroc_ci_upper'],
            'ci_v2_lower': v2_data['auroc_ci_lower'],
            'ci_v2_upper': v2_data['auroc_ci_upper']
        })
    
    comp_df = pd.DataFrame(comparison_records)
    comp_path = output_dir / "h1_v1_vs_v2_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"✓ Saved: {comp_path}")

def main():
    print("="*80)
    print("05_h1_statistical_test_both.py")
    print("H1 가설 검증: Baseline V1 & V2 비교 분석")
    print("="*80)
    
    n_iterations = config.BOOTSTRAP_CONFIG['n_iterations']
    
    # V1 분석
    print("\n[Part 1] Baseline V1 분석")
    print("-"*80)
    results_v1 = load_baseline_results_with_bootstrap(version='v1', 
                                                       n_iterations=n_iterations)
    df_boot_v1 = prepare_bootstrap_data_for_anova(results_v1)
    anova_v1 = one_way_anova_bootstrap(df_boot_v1)
    
    print(f"\nV1 ANOVA: F={anova_v1['f_statistic']:.4f}, "
          f"p={anova_v1['p_value']:.6f}, "
          f"H0 기각={anova_v1['reject_h0']}")
    
    # V2 분석
    print("\n[Part 2] Baseline V2 분석")
    print("-"*80)
    results_v2 = load_baseline_results_with_bootstrap(version='v2', 
                                                       n_iterations=n_iterations)
    df_boot_v2 = prepare_bootstrap_data_for_anova(results_v2)
    anova_v2 = one_way_anova_bootstrap(df_boot_v2)
    
    print(f"\nV2 ANOVA: F={anova_v2['f_statistic']:.4f}, "
          f"p={anova_v2['p_value']:.6f}, "
          f"H0 기각={anova_v2['reject_h0']}")
    
    # Tukey HSD (필요시)
    tukey_v1 = tukey_hsd_posthoc_bootstrap(df_boot_v1) if anova_v1['reject_h0'] else pd.DataFrame()
    tukey_v2 = tukey_hsd_posthoc_bootstrap(df_boot_v2) if anova_v2['reject_h0'] else pd.DataFrame()
    
    # 비교
    print_comparison(results_v1, results_v2, anova_v1, anova_v2)
    
    # 저장
    print("\n[Saving Results]")
    print("-"*80)
    output_dir = config.RESULTS_DIR / "statistics"
    save_results_both(results_v1, results_v2, anova_v1, anova_v2, output_dir)
    
    print("\n" + "="*80)
    print("✓ H1 statistical test (both versions) completed!")
    print(f"✓ Results: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()