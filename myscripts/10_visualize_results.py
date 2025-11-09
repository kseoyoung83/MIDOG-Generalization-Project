#!/usr/bin/env python3
"""
10_visualize_results.py
최종 결과 종합 시각화
- ROC curves
- Confusion matrices
- Training history
- Final report generation
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json

sys.path.append('/workspace/myscripts')
from config import *

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def plot_roc_curves(results_dir, output_dir):
    """ROC curves for all methods"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('ROC Curves by Test Scanner', fontsize=16, fontweight='bold')
    
    scanners = ['Hamamatsu-XR', 'Hamamatsu-S360', 'Aperio-CS2']
    fold_map = {1: 'Hamamatsu-XR', 2: 'Hamamatsu-S360', 3: 'Aperio-CS2'}
    
    for idx, fold in enumerate([1, 2, 3]):
        ax = axes[idx]
        scanner = fold_map[fold]
        
        # Raw baseline
        try:
            pred_file = results_dir / 'metrics' / f'fold{fold}_predictions_v2.csv'
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                fpr, tpr, _ = roc_curve(df['label'], df['prob'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Raw (AUC={roc_auc:.3f})', 
                       linewidth=2, color='#1f77b4')
        except:
            pass
        
        # Normalized methods
        colors = {'macenko': '#ff7f0e', 'vahadane': '#2ca02c', 'reinhard': '#d62728'}
        for method in ['macenko', 'vahadane', 'reinhard']:
            try:
                pred_file = results_dir / 'metrics' / f'fold{fold}_{method}_predictions.csv'
                if pred_file.exists():
                    df = pd.read_csv(pred_file)
                    fpr, tpr, _ = roc_curve(df['true_label'], df['pred_prob'])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{method.capitalize()} (AUC={roc_auc:.3f})', 
                           linewidth=2, color=colors[method])
            except:
                pass
        
        # Random guess
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'Test: {scanner}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved ROC curves")


def plot_confusion_matrices(results_dir, output_dir):
    """Confusion matrices for all methods"""
    
    output_dir = Path(output_dir)
    
    methods = ['raw'] + ['macenko', 'vahadane', 'reinhard']
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Confusion Matrices (Threshold=0.5)', fontsize=16, fontweight='bold')
    
    fold_map = {1: 'Hamamatsu-XR', 2: 'Hamamatsu-S360', 3: 'Aperio-CS2'}
    
    for fold_idx, fold in enumerate([1, 2, 3]):
        for method_idx, method in enumerate(methods):
            ax = axes[fold_idx, method_idx]
            
            try:
                if method == 'raw':
                    pred_file = results_dir / 'metrics' / f'fold{fold}_predictions_v2.csv'
                    if pred_file.exists():
                        df = pd.read_csv(pred_file)
                        y_true = df['label']
                        y_pred = (df['prob'] > 0.5).astype(int)
                else:
                    pred_file = results_dir / 'metrics' / f'fold{fold}_{method}_predictions.csv'
                    if pred_file.exists():
                        df = pd.read_csv(pred_file)
                        y_true = df['true_label']
                        y_pred = df['pred_label']
                
                if pred_file.exists():
                    cm = confusion_matrix(y_true, y_pred)
                    
                    # Normalize
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                               ax=ax, cbar=False, vmin=0, vmax=1,
                               xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
                    
                    if fold_idx == 0:
                        ax.set_title(f'{method.capitalize()}', fontweight='bold')
                    if method_idx == 0:
                        ax.set_ylabel(f'{fold_map[fold]}', fontweight='bold')
                    else:
                        ax.set_ylabel('')
            except Exception as e:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrices")


def plot_training_history(results_dir, output_dir):
    """Training history for baseline and normalized"""
    
    output_dir = Path(output_dir)
    
    # Raw baseline
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History - Raw Baseline', fontsize=16, fontweight='bold')
    
    for fold_idx, fold in enumerate([1, 2, 3]):
        ax = axes[fold_idx]
        
        try:
            history_file = results_dir / 'metrics' / f'fold{fold}_history_v2.csv'
            if history_file.exists():
                df = pd.read_csv(history_file)
                
                ax.plot(df['train_auroc'], label='Train', linewidth=2, color='#1f77b4')
                ax.plot(df['val_auroc'], label='Val', linewidth=2, color='#ff7f0e')
                
                ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
                ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
                ax.set_title(f'Fold {fold}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.3)
        except:
            pass
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history_raw.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized methods
    for method in ['macenko', 'vahadane', 'reinhard']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Training History - {method.capitalize()}', fontsize=16, fontweight='bold')
        
        for fold_idx, fold in enumerate([1, 2, 3]):
            ax = axes[fold_idx]
            
            try:
                history_file = results_dir / 'metrics' / f'fold{fold}_{method}_history.csv'
                if history_file.exists():
                    df = pd.read_csv(history_file)
                    
                    ax.plot(df['train_auroc'], label='Train', linewidth=2, color='#1f77b4')
                    ax.plot(df['val_auroc'], label='Val', linewidth=2, color='#ff7f0e')
                    
                    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
                    ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
                    ax.set_title(f'Fold {fold}', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.3)
            except:
                pass
        
        plt.tight_layout()
        plt.savefig(output_dir / f'training_history_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved training history plots")


def generate_final_report(results_dir, output_dir):
    """Generate final report"""
    
    output_dir = Path(output_dir)
    
    report = []
    report.append("="*80)
    report.append("MIDOG 2021 FINAL REPORT")
    report.append("Stain Normalization for Domain Shift Mitigation")
    report.append("="*80)
    report.append("")
    
    # Load results
    try:
        df_raw = pd.read_csv(results_dir / 'metrics' / 'baseline_summary_v2.csv')
        df_norm = pd.read_csv(results_dir / 'metrics' / 'normalized_summary.csv')
        
        report.append("1. PERFORMANCE SUMMARY")
        report.append("-"*80)
        report.append("")
        report.append(f"{'Method':<15} {'Mean AUROC':<15} {'Std AUROC':<15} {'Mean AUPRC':<15}")
        report.append("-"*80)
        
        raw_auroc = df_raw['test_auroc'].mean()
        raw_auroc_std = df_raw['test_auroc'].std()
        raw_auprc = df_raw['test_auprc'].mean()
        report.append(f"{'Raw Baseline':<15} {raw_auroc:<15.4f} {raw_auroc_std:<15.4f} {raw_auprc:<15.4f}")
        
        for method in ['macenko', 'vahadane', 'reinhard']:
            df_method = df_norm[df_norm['method'] == method]
            auroc = df_method['test_auroc'].mean()
            auroc_std = df_method['test_auroc'].std()
            auprc = df_method['test_auprc'].mean()
            report.append(f"{method.capitalize():<15} {auroc:<15.4f} {auroc_std:<15.4f} {auprc:<15.4f}")
        
        report.append("")
        report.append("2. SCANNER-SPECIFIC RESULTS")
        report.append("-"*80)
        report.append("")
        
        scanners = ['Hamamatsu-XR', 'Hamamatsu-S360', 'Aperio-CS2']
        for scanner_idx, scanner in enumerate(scanners):
            fold = scanner_idx + 1
            report.append(f"Test Scanner: {scanner} (Fold {fold})")
            report.append("")
            
            # Raw
            raw_score = df_raw[df_raw['fold'] == fold]['test_auroc'].values[0]
            report.append(f"  Raw:      {raw_score:.4f}")
            
            # Normalized
            for method in ['macenko', 'vahadane', 'reinhard']:
                df_method = df_norm[(df_norm['method'] == method) & (df_norm['fold'] == fold)]
                if len(df_method) > 0:
                    score = df_method['test_auroc'].values[0]
                    diff = score - raw_score
                    report.append(f"  {method.capitalize():<8}: {score:.4f} ({diff:+.4f})")
            report.append("")
        
        report.append("3. CONCLUSION")
        report.append("-"*80)
        report.append("")
        
        best_method = df_norm.groupby('method')['test_auroc'].mean().idxmax()
        best_auroc = df_norm[df_norm['method'] == best_method]['test_auroc'].mean()
        improvement = best_auroc - raw_auroc
        
        if improvement > 0.05:
            report.append(f"✓ Stain normalization effective: {best_method.capitalize()} +{improvement:.4f}")
        elif improvement > 0.02:
            report.append(f"△ Weak improvement: {best_method.capitalize()} +{improvement:.4f}")
        else:
            report.append(f"✗ No meaningful improvement: best +{improvement:.4f}")
            report.append("")
            report.append("Key Findings:")
            report.append("- Scanner-induced domain shift confirmed (H1 accepted)")
            report.append("- Stain normalization ineffective (H2 rejected)")
            report.append("- Color information alone insufficient")
            report.append("- Feature-level domain adaptation needed")
        
    except Exception as e:
        report.append(f"ERROR: {e}")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'FINAL_REPORT.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)


def main():
    print("="*60)
    print("Final Visualization & Report Generation")
    print("="*60)
    
    results_dir = RESULTS_DIR
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Plotting ROC curves...")
    try:
        plot_roc_curves(results_dir, figures_dir)
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n2. Plotting confusion matrices...")
    try:
        plot_confusion_matrices(results_dir, figures_dir)
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n3. Plotting training history...")
    try:
        plot_training_history(results_dir, figures_dir)
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n4. Generating final report...")
    try:
        generate_final_report(results_dir, results_dir)
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Figures saved to: {figures_dir}")
    print(f"Report saved to: {results_dir / 'FINAL_REPORT.txt'}")


if __name__ == '__main__':
    main()