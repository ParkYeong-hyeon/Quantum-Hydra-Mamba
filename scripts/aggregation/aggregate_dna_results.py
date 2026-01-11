#!/usr/bin/env python3
"""
Aggregate DNA Sequence Classification Results Across All Seeds
Computes mean ± std for all 6 models from 5 seeds
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RESULTS_DIR = Path("./experiments/dna_results")
MODELS = [
    'quantum_hydra',
    'quantum_hydra_hybrid',
    'quantum_mamba',
    'quantum_mamba_hybrid',
    'classical_hydra',
    'classical_mamba'
]
SEEDS = [2024, 2025, 2026, 2027, 2028]

MODEL_DISPLAY_NAMES = {
    'quantum_hydra': 'Quantum Hydra (Superposition)',
    'quantum_hydra_hybrid': 'Quantum Hydra (Hybrid)',
    'quantum_mamba': 'Quantum Mamba (Superposition)',
    'quantum_mamba_hybrid': 'Quantum Mamba (Hybrid)',
    'classical_hydra': 'Classical Hydra',
    'classical_mamba': 'Classical Mamba'
}


def load_results(model_name, seed):
    """Load results JSON for a specific model and seed."""
    results_file = RESULTS_DIR / f"{model_name}_seed{seed}_results.json"
    if not results_file.exists():
        print(f"  WARNING: Missing results for {model_name} seed={seed}")
        return None

    with open(results_file, 'r') as f:
        return json.load(f)


def aggregate_metrics():
    """Aggregate metrics across all seeds for each model."""
    print("="*80)
    print("Aggregating DNA Sequence Classification Results")
    print("="*80)

    aggregated_data = []

    for model_name in MODELS:
        print(f"\nProcessing: {MODEL_DISPLAY_NAMES[model_name]}")

        # Collect metrics across seeds
        test_accs = []
        test_aucs = []
        test_f1s = []
        training_times = []
        n_params = None

        for seed in SEEDS:
            results = load_results(model_name, seed)
            if results is None:
                continue

            test_accs.append(results['test_acc'])
            test_aucs.append(results['test_auc'])
            test_f1s.append(results['test_f1'])
            training_times.append(results['training_time'])

            if n_params is None:
                n_params = results['n_params']

        if len(test_accs) == 0:
            print(f"  ERROR: No results found for {model_name}")
            continue

        # Compute statistics
        acc_mean = np.mean(test_accs)
        acc_std = np.std(test_accs, ddof=1)  # Sample std
        acc_ci = stats.t.interval(0.95, len(test_accs)-1,
                                  loc=acc_mean,
                                  scale=stats.sem(test_accs))

        auc_mean = np.mean(test_aucs)
        auc_std = np.std(test_aucs, ddof=1)

        f1_mean = np.mean(test_f1s)
        f1_std = np.std(test_f1s, ddof=1)

        time_mean = np.mean(training_times)
        time_std = np.std(training_times, ddof=1)

        print(f"  Test Accuracy:  {acc_mean:.4f} ± {acc_std:.4f} (95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}])")
        print(f"  Test AUC:       {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"  Test F1:        {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  Training Time:  {time_mean/60:.2f} ± {time_std/60:.2f} min")
        print(f"  Parameters:     {n_params:,}")
        print(f"  Seeds found:    {len(test_accs)}/{len(SEEDS)}")

        aggregated_data.append({
            'model': MODEL_DISPLAY_NAMES[model_name],
            'model_key': model_name,
            'n_params': n_params,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'acc_ci_lower': acc_ci[0],
            'acc_ci_upper': acc_ci[1],
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'time_mean': time_mean/60,  # Convert to minutes
            'time_std': time_std/60,
            'n_seeds': len(test_accs),
            'individual_accs': test_accs,
            'individual_aucs': test_aucs,
            'individual_f1s': test_f1s
        })

    return aggregated_data


def perform_statistical_tests(aggregated_data):
    """Perform pairwise statistical significance tests."""
    print("\n" + "="*80)
    print("Statistical Significance Tests (Paired t-test)")
    print("="*80)

    # Extract individual accuracy scores
    acc_dict = {d['model_key']: d['individual_accs'] for d in aggregated_data}

    print("\nPairwise comparisons (test accuracy):")
    print("-" * 80)

    for i, data1 in enumerate(aggregated_data):
        for data2 in aggregated_data[i+1:]:
            model1 = data1['model_key']
            model2 = data2['model_key']

            acc1 = np.array(acc_dict[model1])
            acc2 = np.array(acc_dict[model2])

            # Only compare if same number of seeds
            if len(acc1) == len(acc2):
                t_stat, p_value = stats.ttest_rel(acc1, acc2)
                significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

                print(f"{MODEL_DISPLAY_NAMES[model1]:<40} vs {MODEL_DISPLAY_NAMES[model2]:<40}")
                print(f"  t-statistic: {t_stat:8.4f}, p-value: {p_value:.4f} {significant}")


def save_summary_table(aggregated_data):
    """Save aggregated results to CSV and LaTeX table."""
    print("\n" + "="*80)
    print("Saving Summary Tables")
    print("="*80)

    # Create DataFrame
    df = pd.DataFrame([
        {
            'Model': d['model'],
            'Parameters': f"{d['n_params']:,}",
            'Test Accuracy': f"{d['acc_mean']:.4f} ± {d['acc_std']:.4f}",
            'Test AUC': f"{d['auc_mean']:.4f} ± {d['auc_std']:.4f}",
            'Test F1': f"{d['f1_mean']:.4f} ± {d['f1_std']:.4f}",
            'Training Time (min)': f"{d['time_mean']:.2f} ± {d['time_std']:.2f}",
            'Seeds': f"{d['n_seeds']}/{len(SEEDS)}"
        }
        for d in aggregated_data
    ])

    # Sort by test accuracy (descending)
    df_sorted = df.sort_values(by='Test Accuracy', ascending=False, key=lambda x: x.str.split(' ').str[0].astype(float))

    # Save to CSV
    csv_file = RESULTS_DIR / "dna_aggregated_results.csv"
    df_sorted.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")

    # Save to LaTeX
    latex_file = RESULTS_DIR / "dna_aggregated_results.tex"
    latex_table = df_sorted.to_latex(index=False, escape=False)
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Print to console
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)
    print(df_sorted.to_string(index=False))
    print("="*80)


def plot_comparison(aggregated_data):
    """Create comparison plots."""
    print("\n" + "="*80)
    print("Creating Comparison Plots")
    print("="*80)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = [d['model'] for d in aggregated_data]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

    # Plot 1: Test Accuracy with error bars
    ax = axes[0, 0]
    acc_means = [d['acc_mean'] for d in aggregated_data]
    acc_stds = [d['acc_std'] for d in aggregated_data]
    x_pos = np.arange(len(model_names))

    bars = ax.bar(x_pos, acc_means, yerr=acc_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars, acc_means, acc_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Test AUC
    ax = axes[0, 1]
    auc_means = [d['auc_mean'] for d in aggregated_data]
    auc_stds = [d['auc_std'] for d in aggregated_data]

    bars = ax.bar(x_pos, auc_means, yerr=auc_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Test AUC Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    for bar, auc, std in zip(bars, auc_means, auc_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: Training Time
    ax = axes[1, 0]
    time_means = [d['time_mean'] for d in aggregated_data]
    time_stds = [d['time_std'] for d in aggregated_data]

    bars = ax.bar(x_pos, time_means, yerr=time_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bar, time_m, std in zip(bars, time_means, time_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_m:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Accuracy vs Parameters
    ax = axes[1, 1]
    params = [d['n_params'] for d in aggregated_data]

    ax.scatter(params, acc_means, s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    ax.errorbar(params, acc_means, yerr=acc_stds, fmt='none', ecolor='black', alpha=0.5, capsize=5)

    for i, (param, acc, model) in enumerate(zip(params, acc_means, model_names)):
        ax.annotate(model, (param, acc), xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

    ax.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Model Size', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = RESULTS_DIR / "dna_comparison_plots.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_file}")
    plt.close()


def main():
    """Main aggregation pipeline."""
    print("\n" + "="*80)
    print("DNA SEQUENCE CLASSIFICATION RESULTS AGGREGATION")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Models: {len(MODELS)}")
    print(f"Seeds: {len(SEEDS)}")
    print("="*80 + "\n")

    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    # Aggregate metrics
    aggregated_data = aggregate_metrics()

    if len(aggregated_data) == 0:
        print("\nERROR: No results found to aggregate!")
        return

    # Statistical tests
    perform_statistical_tests(aggregated_data)

    # Save summary table
    save_summary_table(aggregated_data)

    # Create plots
    plot_comparison(aggregated_data)

    print("\n" + "="*80)
    print("AGGREGATION COMPLETE!")
    print("="*80)
    print(f"CSV: {RESULTS_DIR}/dna_aggregated_results.csv")
    print(f"LaTeX: {RESULTS_DIR}/dna_aggregated_results.tex")
    print(f"Plots: {RESULTS_DIR}/dna_comparison_plots.pdf")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
