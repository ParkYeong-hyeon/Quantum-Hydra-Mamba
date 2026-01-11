#!/usr/bin/env python3
"""
Aggregate and Analyze Synthetic Benchmark Results

This script:
1. Collects all JSON result files from synthetic benchmark experiments
2. Aggregates results by model, task, and sequence length
3. Computes mean ± std across seeds
4. Generates summary tables and visualizations
5. Performs statistical analysis

Usage:
    python aggregate_synthetic_results.py
    python aggregate_synthetic_results.py --results-dir ./results/synthetic_benchmarks
    python aggregate_synthetic_results.py --output-dir ./analysis/synthetic
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Skipping plots.")


# Model information
MODEL_INFO = {
    '1a': {'name': 'QuantumTransformer', 'group': 1, 'type': 'transformer'},
    '1b': {'name': 'QuantumMambaSSM', 'group': 1, 'type': 'mamba'},
    '1c': {'name': 'QuantumHydraSSM', 'group': 1, 'type': 'hydra'},
    '2a': {'name': 'ClassicalQuantumAttention', 'group': 2, 'type': 'transformer'},
    '2d': {'name': 'QuantumMambaHydraSSM', 'group': 2, 'type': 'mamba'},
    '2e': {'name': 'QuantumHydraHydraSSM', 'group': 2, 'type': 'hydra'},
    '3a': {'name': 'ClassicalTransformer', 'group': 3, 'type': 'transformer'},
    '3b': {'name': 'TrueClassicalMamba', 'group': 3, 'type': 'mamba'},
    '3c': {'name': 'TrueClassicalHydra', 'group': 3, 'type': 'hydra'},
    '4a': {'name': 'QuantumTransformerE2E', 'group': 4, 'type': 'transformer'},
    '4d': {'name': 'QuantumMambaE2E_Super', 'group': 4, 'type': 'mamba'},
    '4e': {'name': 'QuantumHydraE2E_Super', 'group': 4, 'type': 'hydra'},
}

GROUP_NAMES = {
    1: 'Q-Feat + C-Mix',
    2: 'C-Feat + Q-Mix',
    3: 'Classical',
    4: 'E2E Quantum'
}


def load_results(results_dir):
    """Load all JSON result files."""
    results_dir = Path(results_dir)
    results = []

    for json_file in results_dir.glob("synthetic_*_results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Loaded {len(results)} result files from {results_dir}")
    return results


def aggregate_results(results):
    """Aggregate results by model, task, and sequence length."""
    # Group by (model_id, task, seq_len)
    grouped = defaultdict(list)

    for r in results:
        key = (r['model_id'], r['task'], r['seq_len'])
        grouped[key].append(r)

    # Compute statistics
    aggregated = []

    for (model_id, task, seq_len), runs in grouped.items():
        n_seeds = len(runs)

        if runs[0]['task_type'] == 'classification':
            # Classification metrics
            accs = [r['test_acc'] for r in runs]
            aucs = [r['test_auc'] for r in runs]
            f1s = [r['test_f1'] for r in runs]

            agg = {
                'model_id': model_id,
                'model_name': MODEL_INFO[model_id]['name'],
                'group': MODEL_INFO[model_id]['group'],
                'group_name': GROUP_NAMES[MODEL_INFO[model_id]['group']],
                'arch_type': MODEL_INFO[model_id]['type'],
                'task': task,
                'task_type': 'classification',
                'seq_len': seq_len,
                'n_seeds': n_seeds,
                'test_acc_mean': np.mean(accs),
                'test_acc_std': np.std(accs),
                'test_auc_mean': np.mean(aucs),
                'test_auc_std': np.std(aucs),
                'test_f1_mean': np.mean(f1s),
                'test_f1_std': np.std(f1s),
                'baseline': runs[0]['baseline'],
            }
        else:
            # Regression metrics
            mses = [r['test_mse'] for r in runs]
            maes = [r['test_mae'] for r in runs]
            r2s = [r['test_r2'] for r in runs]

            agg = {
                'model_id': model_id,
                'model_name': MODEL_INFO[model_id]['name'],
                'group': MODEL_INFO[model_id]['group'],
                'group_name': GROUP_NAMES[MODEL_INFO[model_id]['group']],
                'arch_type': MODEL_INFO[model_id]['type'],
                'task': task,
                'task_type': 'regression',
                'seq_len': seq_len,
                'n_seeds': n_seeds,
                'test_mse_mean': np.mean(mses),
                'test_mse_std': np.std(mses),
                'test_mae_mean': np.mean(maes),
                'test_mae_std': np.std(maes),
                'test_r2_mean': np.mean(r2s),
                'test_r2_std': np.std(r2s),
                'baseline': runs[0]['baseline'],
            }

            # Calculate improvement over baseline
            agg['improvement_pct'] = (1 - agg['test_mse_mean'] / agg['baseline']) * 100

        # Common fields
        times = [r['training_time'] for r in runs]
        agg['training_time_mean'] = np.mean(times)
        agg['training_time_std'] = np.std(times)
        agg['n_params'] = runs[0]['n_params']

        aggregated.append(agg)

    return aggregated


def create_summary_tables(aggregated, output_dir):
    """Create summary tables for each task."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(aggregated)

    tables = {}

    for task in df['task'].unique():
        task_df = df[df['task'] == task].copy()
        task_type = task_df['task_type'].iloc[0]

        # Sort by sequence length, then by performance
        if task_type == 'classification':
            task_df = task_df.sort_values(['seq_len', 'test_acc_mean'], ascending=[True, False])
        else:
            task_df = task_df.sort_values(['seq_len', 'test_mse_mean'], ascending=[True, True])

        tables[task] = task_df

        # Save CSV
        csv_path = output_dir / f"summary_{task}.csv"
        task_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    return tables


def generate_markdown_report(aggregated, output_dir):
    """Generate a markdown report summarizing all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(aggregated)

    report_lines = []
    report_lines.append("# Synthetic Benchmark Results Summary\n")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Total Experiments:** {len(aggregated)}\n")
    report_lines.append("\n---\n")

    # Overview
    report_lines.append("## Overview\n")
    report_lines.append(f"- Models tested: {df['model_id'].nunique()}\n")
    report_lines.append(f"- Tasks: {', '.join(df['task'].unique())}\n")
    report_lines.append(f"- Sequence lengths: {sorted(df['seq_len'].unique())}\n")
    report_lines.append("\n")

    # Results by task
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task]
        task_type = task_df['task_type'].iloc[0]

        report_lines.append(f"## {task.replace('_', ' ').title()}\n")
        report_lines.append(f"**Task type:** {task_type}\n")
        report_lines.append(f"**Baseline:** {task_df['baseline'].iloc[0]:.4f}\n\n")

        for seq_len in sorted(task_df['seq_len'].unique()):
            len_df = task_df[task_df['seq_len'] == seq_len].copy()

            if task_type == 'classification':
                len_df = len_df.sort_values('test_acc_mean', ascending=False)
                report_lines.append(f"### Sequence Length: {seq_len}\n\n")
                report_lines.append("| Rank | Model | Group | Test Acc | Test AUC | Time (s) |\n")
                report_lines.append("|------|-------|-------|----------|----------|----------|\n")

                for rank, (_, row) in enumerate(len_df.iterrows(), 1):
                    report_lines.append(
                        f"| {rank} | {row['model_id']} {row['model_name']} | {row['group_name']} | "
                        f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f} | "
                        f"{row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f} | "
                        f"{row['training_time_mean']:.1f} |\n"
                    )
            else:
                len_df = len_df.sort_values('test_mse_mean', ascending=True)
                report_lines.append(f"### Sequence Length: {seq_len}\n\n")
                report_lines.append("| Rank | Model | Group | Test MSE | Test R² | Improvement | Time (s) |\n")
                report_lines.append("|------|-------|-------|----------|---------|-------------|----------|\n")

                for rank, (_, row) in enumerate(len_df.iterrows(), 1):
                    report_lines.append(
                        f"| {rank} | {row['model_id']} {row['model_name']} | {row['group_name']} | "
                        f"{row['test_mse_mean']:.6f} ± {row['test_mse_std']:.6f} | "
                        f"{row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f} | "
                        f"{row['improvement_pct']:.1f}% | "
                        f"{row['training_time_mean']:.1f} |\n"
                    )

            report_lines.append("\n")

    # Key findings
    report_lines.append("---\n")
    report_lines.append("## Key Findings\n\n")

    # Find best models per task
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task]
        task_type = task_df['task_type'].iloc[0]

        if task_type == 'classification':
            best = task_df.loc[task_df['test_acc_mean'].idxmax()]
            report_lines.append(f"- **{task}**: Best model is **{best['model_id']} ({best['model_name']})** "
                              f"with accuracy {best['test_acc_mean']:.4f} ± {best['test_acc_std']:.4f}\n")
        else:
            best = task_df.loc[task_df['test_mse_mean'].idxmin()]
            report_lines.append(f"- **{task}**: Best model is **{best['model_id']} ({best['model_name']})** "
                              f"with MSE {best['test_mse_mean']:.6f} ± {best['test_mse_std']:.6f} "
                              f"({best['improvement_pct']:.1f}% improvement over baseline)\n")

    # Save report
    report_path = output_dir / "SYNTHETIC_BENCHMARK_RESULTS.md"
    with open(report_path, 'w') as f:
        f.writelines(report_lines)

    print(f"Saved: {report_path}")
    return report_path


def plot_results(aggregated, output_dir):
    """Generate visualization plots."""
    if not HAS_PLOTTING:
        print("Skipping plots (matplotlib/seaborn not available)")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(aggregated)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        task_type = task_df['task_type'].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 6))

        if task_type == 'classification':
            metric = 'test_acc_mean'
            ylabel = 'Test Accuracy'
            title = f'{task.replace("_", " ").title()}: Accuracy vs Sequence Length'
        else:
            metric = 'test_mse_mean'
            ylabel = 'Test MSE (log scale)'
            title = f'{task.replace("_", " ").title()}: MSE vs Sequence Length'
            ax.set_yscale('log')

        # Plot each model
        for model_id in sorted(task_df['model_id'].unique()):
            model_df = task_df[task_df['model_id'] == model_id].sort_values('seq_len')

            group = MODEL_INFO[model_id]['group']
            color = plt.cm.tab10(group - 1)

            ax.errorbar(
                model_df['seq_len'],
                model_df[metric],
                yerr=model_df[metric.replace('mean', 'std')],
                label=f"{model_id} ({MODEL_INFO[model_id]['name']})",
                marker='o',
                capsize=3,
                color=color
            )

        # Add baseline
        baseline = task_df['baseline'].iloc[0]
        ax.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.4f})')

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        plot_path = output_dir / f"plot_{task}_vs_seqlen.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")

    # Heatmap of model performance
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        task_type = task_df['task_type'].iloc[0]

        if task_type == 'classification':
            metric = 'test_acc_mean'
            cmap = 'RdYlGn'
        else:
            metric = 'test_mse_mean'
            cmap = 'RdYlGn_r'

        # Pivot for heatmap
        pivot = task_df.pivot(index='model_id', columns='seq_len', values=metric)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap=cmap, ax=ax)
        ax.set_title(f'{task.replace("_", " ").title()}: Model Performance Heatmap')

        plt.tight_layout()

        plot_path = output_dir / f"heatmap_{task}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze synthetic benchmark results"
    )

    parser.add_argument("--results-dir", type=str, default="./results/synthetic_benchmarks",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=str, default="./analysis/synthetic_benchmarks",
                        help="Output directory for analysis")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")

    args = parser.parse_args()

    print("=" * 60)
    print("Aggregating Synthetic Benchmark Results")
    print("=" * 60)

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("No results found. Exiting.")
        return

    # Aggregate
    aggregated = aggregate_results(results)
    print(f"Aggregated {len(aggregated)} unique (model, task, seq_len) combinations")

    # Create summary tables
    tables = create_summary_tables(aggregated, args.output_dir)

    # Generate markdown report
    report_path = generate_markdown_report(aggregated, args.output_dir)

    # Generate plots
    if not args.no_plots:
        plot_results(aggregated, args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Main report: {report_path}")


if __name__ == "__main__":
    main()
