#!/usr/bin/env python3
"""
09_compare_results.py
Raw baseline vs Normalized 비교 분석
- 기술 통계
- 시각적 비교
- 간단한 효과 크기 계산
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/workspace/myscripts')
from config import *

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def load_results():
    """결과 파일 로드"""
    
    results_dir = RESULTS_DIR / 'metrics'
    
    # Raw baseline
    raw_v2 = results_dir / 'baseline_summary_v2.csv'
    
    # Normalized
    normalized = results_dir / 'normalized_summary.csv'
    
    if not raw_v2.exists():
        print(f"ERROR: {raw_v2} not found!")
        return None, None
    
    if not normalized.exists():
        print(f"ERROR: {normalized} not found!")
        return None, None
    
    df_raw = pd.read_csv(raw_v2)
    df_norm = pd.read_csv(normalized)
    
    print(f"✓ Loaded Raw: {len(df_raw)} results")
    print(f"✓ Loaded Normalized: {len(df_norm)} results")
    
    return df_raw, df_norm


def compute_statistics(df_raw, df_norm):
    """기술 통계 계산"""
    
    stats = {
        'metric': [],
        'raw_mean': [],
        'raw_std': [],
        'macenko_mean': [],
        'macenko_std': [],
        'vahadane_mean': [],
        'vahadane_std': [],
        'reinhard_mean': [],
        'reinhard_std': []
    }
    
    # AUROC
    stats['metric'].append('AUROC')
    stats['raw_mean'].append(df_raw['test_auroc'].mean())
    stats['raw_std'].append(df_raw['test_auroc'].std())
    
    for method in ['macenko', 'vahadane', 'reinhard']:
        df_method = df_norm[df_norm['method'] == method]
        stats[f'{method}_mean'].append(df_method['test_auroc'].mean())
        stats[f'{method}_std'].append(df_method['test_auroc'].std())
    
    # AUPRC
    stats['metric'].append('AUPRC')
    stats['raw_mean'].append(df_raw['test_auprc'].mean())
    stats['raw_std'].append(df_raw['test_auprc'].std())
    
    for method in ['macenko', 'vahadane', 'reinhard']:
        df_method = df_norm[df_norm['method'] == method]
        stats[f'{method}_mean'].append(df_method['test_auprc'].mean())
        stats[f'{method}_std'].append(df_method['test_auprc'].std())
    
    df_stats = pd.DataFrame(stats)
    return df_stats


def compute_effect_size(df_raw, df_norm):
    """Cohen's d 효과 크기 계산"""
    
    results = []
    
    for method in ['macenko', 'vahadane', 'reinhard']:
        df_method = df_norm[df_norm['method'] == method]
        
        # AUROC
        raw_auroc = df_raw['test_auroc'].values
        norm_auroc = df_method['test_auroc'].values
        
        # Cohen's d
        mean_diff = norm_auroc.mean() - raw_auroc.mean()
        pooled_std = np.sqrt((raw_auroc.std()**2 + norm_auroc.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'method': method,
            'auroc_diff': mean_diff,
            'cohens_d': cohens_d,
            'interpretation': interpret_cohens_d(cohens_d)
        })
    
    return pd.DataFrame(results)


def interpret_cohens_d(d):
    """Cohen's d 해석"""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'Negligible'
    elif d_abs < 0.5:
        return 'Small'
    elif d_abs < 0.8:
        return 'Medium'
    else:
        return 'Large'


def create_comparison_plots(df_raw, df_norm, output_dir):
    """비교 시각화"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar plot: Mean AUROC
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Raw', 'Macenko', 'Vahadane', 'Reinhard']
    means = [df_raw['test_auroc'].mean()]
    stds = [df_raw['test_auroc'].std()]
    
    for method in ['macenko', 'vahadane', 'reinhard']:
        df_method = df_norm[df_norm['method'] == method]
        means.append(df_method['test_auroc'].mean())
        stds.append(df_method['test_auroc'].std())
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Test AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Raw vs Stain Normalized Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0.45, 0.55)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random guess')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot: AUROC distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    labels = []
    
    data.append(df_raw['test_auroc'].values)
    labels.append('Raw')
    
    for method in ['macenko', 'vahadane', 'reinhard']:
        df_method = df_norm[df_norm['method'] == method]
        data.append(df_method['test_auroc'].values)
        labels.append(method.capitalize())
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax.set_ylabel('Test AUROC', fontsize=12, fontweight='bold')
    ax.set_title('AUROC Distribution by Method', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random guess')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scanner-specific comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scanners = df_raw['test_scanner'].unique() if 'test_scanner' in df_raw.columns else \
               ['Hamamatsu-XR', 'Hamamatsu-S360', 'Aperio-CS2']
    
    x = np.arange(len(scanners))
    width = 0.2
    
    # Raw
    raw_scores = []
    for scanner in scanners:
        if 'test_scanner' in df_raw.columns:
            score = df_raw[df_raw['test_scanner'] == scanner]['test_auroc'].values[0]
        else:
            # Assume fold order matches scanner order
            fold_idx = list(scanners).index(scanner) + 1
            score = df_raw[df_raw['fold'] == fold_idx]['test_auroc'].values[0]
        raw_scores.append(score)
    
    ax.bar(x - 1.5*width, raw_scores, width, label='Raw', color='#1f77b4', alpha=0.7)
    
    # Normalized methods
    colors = {'macenko': '#ff7f0e', 'vahadane': '#2ca02c', 'reinhard': '#d62728'}
    for i, method in enumerate(['macenko', 'vahadane', 'reinhard']):
        scores = []
        for scanner in scanners:
            df_method = df_norm[df_norm['method'] == method]
            if 'test_scanner' in df_method.columns:
                score = df_method[df_method['test_scanner'] == scanner]['test_auroc'].values[0]
            else:
                fold_idx = list(scanners).index(scanner) + 1
                score = df_method[df_method['fold'] == fold_idx]['test_auroc'].values[0]
            scores.append(score)
        
        ax.bar(x + (i-0.5)*width, scores, width, label=method.capitalize(), 
               color=colors[method], alpha=0.7)
    
    ax.set_ylabel('Test AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Test Scanner', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scanners, fontsize=10, rotation=15, ha='right')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random guess')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_by_scanner.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 3 plots to {output_dir}")


def main():
    print("="*60)
    print("Raw vs Normalized Comparison")
    print("="*60)
    
    # Load results
    df_raw, df_norm = load_results()
    if df_raw is None or df_norm is None:
        return
    
    # Compute statistics
    print("\n1. Computing statistics...")
    df_stats = compute_statistics(df_raw, df_norm)
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    print(df_stats.to_string(index=False))
    
    # Effect size
    print("\n2. Computing effect sizes...")
    df_effect = compute_effect_size(df_raw, df_norm)
    print("\n" + "="*60)
    print("EFFECT SIZE (Cohen's d)")
    print("="*60)
    print(df_effect.to_string(index=False))
    
    # Save results
    stats_dir = RESULTS_DIR / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    df_stats.to_csv(stats_dir / 'comparison_statistics.csv', index=False)
    df_effect.to_csv(stats_dir / 'comparison_effect_size.csv', index=False)
    
    # Create plots
    print("\n3. Creating visualizations...")
    figures_dir = RESULTS_DIR / 'figures'
    create_comparison_plots(df_raw, df_norm, figures_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Raw Baseline:  AUROC {df_raw['test_auroc'].mean():.4f} ± {df_raw['test_auroc'].std():.4f}")
    print(f"Macenko:       AUROC {df_norm[df_norm['method']=='macenko']['test_auroc'].mean():.4f} ± {df_norm[df_norm['method']=='macenko']['test_auroc'].std():.4f}")
    print(f"Vahadane:      AUROC {df_norm[df_norm['method']=='vahadane']['test_auroc'].mean():.4f} ± {df_norm[df_norm['method']=='vahadane']['test_auroc'].std():.4f}")
    print(f"Reinhard:      AUROC {df_norm[df_norm['method']=='reinhard']['test_auroc'].mean():.4f} ± {df_norm[df_norm['method']=='reinhard']['test_auroc'].std():.4f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    raw_mean = df_raw['test_auroc'].mean()
    best_method = df_norm.groupby('method')['test_auroc'].mean().idxmax()
    best_mean = df_norm[df_norm['method']==best_method]['test_auroc'].mean()
    improvement = best_mean - raw_mean
    
    if improvement > 0.05:
        print(f"✓ {best_method.capitalize()} shows meaningful improvement (+{improvement:.4f})")
        print("  → Stain normalization has positive effect")
    elif improvement > 0.02:
        print(f"△ {best_method.capitalize()} shows slight improvement (+{improvement:.4f})")
        print("  → Weak effect of stain normalization")
    else:
        print(f"✗ No meaningful improvement (best: +{improvement:.4f})")
        print("  → Stain normalization ineffective")
        print("  → Color information alone insufficient")
        print("  → Feature-level domain adaptation needed")
    
    print("\n✓ Analysis complete!")
    print(f"   Results: {stats_dir}")
    print(f"   Figures: {figures_dir}")


if __name__ == '__main__':
    main()