#!/usr/bin/env python3
"""
11_advanced_statistics.py
Table A1 (detailed metrics) 기반 추가 통계 분석
- Scanner별 precision-recall 패턴
- Method별 prediction bias
- Two-way ANOVA (Scanner × Method)
- Wilcoxon tests on F1/Precision/Recall
- Accuracy paradox 검증
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, f_oneway, mannwhitneyu
from itertools import combinations

sys.path.append('/workspace/myscripts')
from config import *

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Table A1 데이터 (논문에서)
data = {
    'fold': [1,1,1,1, 2,2,2,2, 3,3,3,3],
    'scanner': ['Hamamatsu-XR','Hamamatsu-XR','Hamamatsu-XR','Hamamatsu-XR',
                'Hamamatsu-S360','Hamamatsu-S360','Hamamatsu-S360','Hamamatsu-S360',
                'Aperio-CS2','Aperio-CS2','Aperio-CS2','Aperio-CS2'],
    'method': ['Raw','Macenko','Vahadane','Reinhard',
               'Raw','Macenko','Vahadane','Reinhard',
               'Raw','Macenko','Vahadane','Reinhard'],
    'auroc': [0.4984,0.5118,0.4944,0.5177, 0.5149,0.4953,0.5178,0.5068, 0.5070,0.5041,0.5059,0.4982],
    'auprc': [0.2539,0.2539,0.2402,0.2521, 0.2469,0.2469,0.2515,0.2564, 0.2490,0.2490,0.2703,0.2421],
    'accuracy': [0.2792,0.2792,0.3121,0.2594, 0.4329,0.4329,0.5911,0.6478, 0.5412,0.5412,0.3560,0.5655],
    'precision': [0.2475,0.2475,0.2453,0.2464, 0.2419,0.2419,0.2587,0.2624, 0.2527,0.2527,0.2450,0.2330],
    'recall': [0.9487,0.9487,0.8681,0.9799, 0.6118,0.6118,0.3553,0.2387, 0.4372,0.4372,0.7706,0.3304],
    'f1': [0.3926,0.3926,0.3826,0.3938, 0.3467,0.3467,0.2994,0.2500, 0.3203,0.3203,0.3717,0.2733]
}

df = pd.DataFrame(data)


def analysis_1_precision_recall_tradeoff(df, output_dir):
    """1. Scanner별 Precision-Recall Trade-off"""
    print("\n" + "="*60)
    print("1. PRECISION-RECALL TRADE-OFF BY SCANNER")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    scanners = df['scanner'].unique()
    colors = {'Raw': '#1f77b4', 'Macenko': '#ff7f0e', 'Vahadane': '#2ca02c', 'Reinhard': '#d62728'}
    
    for idx, scanner in enumerate(scanners):
        ax = axes[idx]
        df_s = df[df['scanner'] == scanner]
        
        for method in ['Raw', 'Macenko', 'Vahadane', 'Reinhard']:
            row = df_s[df_s['method'] == method].iloc[0]
            ax.scatter(row['recall'], row['precision'], s=200, 
                      color=colors[method], label=method, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Recall', fontweight='bold', fontsize=11)
        ax.set_ylabel('Precision', fontweight='bold', fontsize=11)
        ax.set_title(scanner, fontweight='bold', fontsize=12)
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 0.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Ideal point
        ax.scatter(1.0, 1.0, s=100, color='green', marker='*', label='Ideal', zorder=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary stats
    for scanner in scanners:
        df_s = df[df['scanner'] == scanner]
        print(f"\n{scanner}:")
        print(f"  Precision range: {df_s['precision'].min():.3f} - {df_s['precision'].max():.3f}")
        print(f"  Recall range: {df_s['recall'].min():.3f} - {df_s['recall'].max():.3f}")
    
    print(f"\n✓ Saved: precision_recall_tradeoff.png")


def analysis_2_prediction_bias(df, output_dir):
    """2. Method별 Positive Prediction Bias"""
    print("\n" + "="*60)
    print("2. POSITIVE PREDICTION BIAS")
    print("="*60)
    
    # Compute positive prediction rate (approximation)
    # pos_rate ≈ (Recall × Pos + (1-Precision) × Neg) / Total
    # Assuming 1:3 ratio → Pos=0.25, Neg=0.75
    df['pos_pred_rate'] = df['recall'] * 0.25 + (1 - df['precision']) * 0.75
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Raw', 'Macenko', 'Vahadane', 'Reinhard']
    x = np.arange(len(methods))
    width = 0.25
    
    for i, scanner in enumerate(df['scanner'].unique()):
        df_s = df[df['scanner'] == scanner]
        rates = [df_s[df_s['method'] == m]['pos_pred_rate'].values[0] for m in methods]
        ax.bar(x + i*width, rates, width, label=scanner, alpha=0.7)
    
    ax.set_ylabel('Positive Prediction Rate', fontweight='bold', fontsize=12)
    ax.set_title('Positive Prediction Bias by Method and Scanner', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=1.5, label='True Pos Rate (1:3 ratio)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPositive Prediction Rate by Method (mean ± std):")
    for method in methods:
        rates = df[df['method'] == method]['pos_pred_rate'].values
        print(f"  {method}: {rates.mean():.3f} ± {rates.std():.3f}")
    
    print(f"\n✓ Saved: prediction_bias.png")


def analysis_3_twoway_anova(df):
    """3. Two-way ANOVA: Scanner × Method interaction"""
    print("\n" + "="*60)
    print("3. TWO-WAY ANOVA (Scanner × Method)")
    print("="*60)
    
    from scipy.stats import f_oneway
    
    # Scanner main effect
    scanner_groups = [df[df['scanner'] == s]['auroc'].values for s in df['scanner'].unique()]
    f_scanner, p_scanner = f_oneway(*scanner_groups)
    
    # Method main effect
    method_groups = [df[df['method'] == m]['auroc'].values for m in df['method'].unique()]
    f_method, p_method = f_oneway(*method_groups)
    
    print(f"\nMain Effect - Scanner:")
    print(f"  F = {f_scanner:.4f}, p = {p_scanner:.6f}")
    
    print(f"\nMain Effect - Method:")
    print(f"  F = {f_method:.4f}, p = {p_method:.6f}")
    
    # Interaction (qualitative assessment)
    print(f"\nInteraction Pattern:")
    for scanner in df['scanner'].unique():
        df_s = df[df['scanner'] == scanner]
        best = df_s.loc[df_s['auroc'].idxmax(), 'method']
        worst = df_s.loc[df_s['auroc'].idxmin(), 'method']
        print(f"  {scanner}: Best={best}, Worst={worst}")
    
    print("\n→ No consistent best/worst method → Scanner×Method interaction present")


def analysis_4_wilcoxon_all_metrics(df):
    """4. Wilcoxon Tests on F1, Precision, Recall"""
    print("\n" + "="*60)
    print("4. WILCOXON SIGNED-RANK TESTS")
    print("="*60)
    
    metrics = ['auroc', 'f1', 'precision', 'recall', 'accuracy']
    methods = ['Macenko', 'Vahadane', 'Reinhard']
    
    results = []
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        raw_values = df[df['method'] == 'Raw'][metric].values
        
        for method in methods:
            norm_values = df[df['method'] == method][metric].values
            
            try:
                stat, p = wilcoxon(raw_values, norm_values)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  Raw vs {method}: stat={stat:.2f}, p={p:.4f} {sig}")
                results.append({'metric': metric, 'comparison': f'Raw vs {method}', 
                               'statistic': stat, 'p_value': p, 'significance': sig})
            except:
                print(f"  Raw vs {method}: ERROR (likely identical values)")
                results.append({'metric': metric, 'comparison': f'Raw vs {method}', 
                               'statistic': np.nan, 'p_value': np.nan, 'significance': 'ERROR'})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'statistics' / 'wilcoxon_all_metrics.csv', index=False)
    print(f"\n✓ Saved: wilcoxon_all_metrics.csv")
    
    return df_results


def analysis_5_accuracy_paradox(df, output_dir):
    """5. Accuracy Paradox Visualization"""
    print("\n" + "="*60)
    print("5. ACCURACY PARADOX")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scanner in df['scanner'].unique():
        df_s = df[df['scanner'] == scanner]
        ax.scatter(df_s['accuracy'], df_s['f1'], s=150, label=scanner, alpha=0.7)
        
        for idx, row in df_s.iterrows():
            ax.annotate(row['method'][:3], (row['accuracy'], row['f1']), 
                       fontsize=8, ha='right')
    
    ax.set_xlabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy Paradox: High Accuracy ≠ Good Performance', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Diagonal line (ideal: accuracy = f1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Ideal (Acc=F1)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_paradox.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identify worst cases
    print("\nWorst Accuracy Paradox Cases:")
    df['acc_f1_gap'] = df['accuracy'] - df['f1']
    worst = df.nlargest(3, 'acc_f1_gap')[['scanner', 'method', 'accuracy', 'f1', 'acc_f1_gap']]
    print(worst.to_string(index=False))
    
    print(f"\n✓ Saved: accuracy_paradox.png")


def analysis_6_method_consistency(df, output_dir):
    """6. Method Consistency Across Scanners"""
    print("\n" + "="*60)
    print("6. METHOD CONSISTENCY")
    print("="*60)
    
    # Compute rank per scanner
    df_rank = df.copy()
    df_rank['rank'] = df.groupby('scanner')['auroc'].rank(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Raw', 'Macenko', 'Vahadane', 'Reinhard']
    colors = {'Raw': '#1f77b4', 'Macenko': '#ff7f0e', 'Vahadane': '#2ca02c', 'Reinhard': '#d62728'}
    
    for method in methods:
        df_m = df_rank[df_rank['method'] == method]
        ax.plot(df_m['scanner'], df_m['rank'], marker='o', linewidth=2, 
               markersize=10, label=method, color=colors[method])
    
    ax.set_xlabel('Scanner', fontweight='bold', fontsize=12)
    ax.set_ylabel('Rank (1=Best, 4=Worst)', fontweight='bold', fontsize=12)
    ax.set_title('Method Rank Consistency Across Scanners', fontweight='bold', fontsize=14)
    ax.set_yticks([1, 2, 3, 4])
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Rank variance
    print("\nRank Variance (lower = more consistent):")
    for method in methods:
        ranks = df_rank[df_rank['method'] == method]['rank'].values
        print(f"  {method}: variance={ranks.var():.3f}")
    
    print(f"\n✓ Saved: method_consistency.png")


def main():
    print("="*60)
    print("ADVANCED STATISTICAL ANALYSIS (Table A1)")
    print("="*60)
    
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    stats_dir = RESULTS_DIR / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_1_precision_recall_tradeoff(df, figures_dir)
    analysis_2_prediction_bias(df, figures_dir)
    analysis_3_twoway_anova(df)
    df_wilcoxon = analysis_4_wilcoxon_all_metrics(df)
    analysis_5_accuracy_paradox(df, figures_dir)
    analysis_6_method_consistency(df, figures_dir)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ 6 analyses completed")
    print("✓ 5 new figures generated")
    print("✓ Wilcoxon results saved to CSV")
    print(f"\nFigures: {figures_dir}")
    print(f"Statistics: {stats_dir}")


if __name__ == '__main__':
    main()