#!/usr/bin/env python3
"""
Aggregate and analyze Forrelation experiment results.

This script processes all result JSON files and creates summary tables
for analyzing quantum advantage according to the test plan.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results(results_dir="experiments/forrelation_results"):
    """Load all result JSON files."""
    results_path = Path(results_dir)
    json_files = list(results_path.glob("*_results.json"))

    print(f"Found {len(json_files)} result files")

    all_results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract key information
            result = {
                'model_name': data['model_name'],
                'seed': data['seed'],
                'n_bits': data['dataset_params']['n_bits'],
                'seq_len': data['dataset_params']['seq_len'],
                'n_params': data['n_params'],
                'best_test_acc': data['best_test_acc'],
                'final_test_acc': data['final_test_acc'],
                'final_test_auc': data['final_test_auc'],
                'final_test_f1': data['final_test_f1'],
                'training_time': data['training_time'],
                'n_epochs_trained': len(data['history']['epochs']),
                'dataset_file': Path(data['dataset_path']).name
            }
            all_results.append(result)

        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    return pd.DataFrame(all_results)


def analyze_sample_efficiency(df, n_bits=6):
    """
    Analyze sample efficiency (Silver Standard).
    Compare quantum vs classical performance across varying sequence lengths.
    """
    print("\n" + "="*80)
    print(f"SAMPLE EFFICIENCY ANALYSIS (n_bits={n_bits})")
    print("="*80)

    # Filter for the specific n_bits
    df_filtered = df[df['n_bits'] == n_bits].copy()

    # Categorize models
    df_filtered['model_type'] = df_filtered['model_name'].apply(
        lambda x: 'Quantum' if 'quantum' in x else 'Classical'
    )

    # Group by model, sequence length, and compute statistics
    summary = df_filtered.groupby(['model_name', 'model_type', 'seq_len']).agg({
        'best_test_acc': ['mean', 'std', 'count'],
        'final_test_auc': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).round(4)

    print("\nPer-model accuracy vs sequence length:")
    print(summary)

    # Compute quantum vs classical comparison
    quantum_summary = df_filtered[df_filtered['model_type'] == 'Quantum'].groupby('seq_len').agg({
        'best_test_acc': 'mean'
    }).rename(columns={'best_test_acc': 'quantum_acc'})

    classical_summary = df_filtered[df_filtered['model_type'] == 'Classical'].groupby('seq_len').agg({
        'best_test_acc': 'mean'
    }).rename(columns={'best_test_acc': 'classical_acc'})

    comparison = quantum_summary.join(classical_summary)
    comparison['advantage'] = comparison['quantum_acc'] - comparison['classical_acc']
    comparison['advantage_pct'] = (comparison['advantage'] / comparison['classical_acc'] * 100).round(2)

    print("\nQuantum vs Classical Comparison:")
    print(comparison)

    # Determine if Silver Standard is met
    print("\nSilver Standard Assessment:")
    target_acc = 0.90  # 90% accuracy threshold

    quantum_L_to_target = None
    classical_L_to_target = None

    for L in sorted(comparison.index):
        if quantum_L_to_target is None and comparison.loc[L, 'quantum_acc'] >= target_acc:
            quantum_L_to_target = L
        if classical_L_to_target is None and comparison.loc[L, 'classical_acc'] >= target_acc:
            classical_L_to_target = L

    if quantum_L_to_target and classical_L_to_target:
        print(f"  Quantum models reach {target_acc:.0%} accuracy at L={quantum_L_to_target}")
        print(f"  Classical models reach {target_acc:.0%} accuracy at L={classical_L_to_target}")
        print(f"  Quantum advantage: {((classical_L_to_target - quantum_L_to_target) / classical_L_to_target * 100):.1f}% fewer samples")

        if quantum_L_to_target < classical_L_to_target:
            print("\n  ✓ SILVER STANDARD MET: Quantum models show sample efficiency advantage!")
        else:
            print("\n  ✗ Silver Standard not met: No clear advantage observed")
    else:
        print(f"  One or more model types did not reach {target_acc:.0%} accuracy")

    return comparison


def analyze_scaling(df):
    """
    Analyze scaling behavior (Gold Standard).
    Compare how the advantage changes with problem complexity (n_bits).
    """
    print("\n" + "="*80)
    print("SCALING ANALYSIS (GOLD STANDARD)")
    print("="*80)

    # Categorize models
    df['model_type'] = df['model_name'].apply(
        lambda x: 'Quantum' if 'quantum' in x else 'Classical'
    )

    # Group by n_bits, seq_len, model_type
    scaling_summary = df.groupby(['n_bits', 'seq_len', 'model_type']).agg({
        'best_test_acc': ['mean', 'std']
    }).round(4)

    print("\nScaling summary:")
    print(scaling_summary)

    # Compute advantage at each complexity level
    advantages = {}
    for n_bits in df['n_bits'].unique():
        df_n = df[df['n_bits'] == n_bits]

        quantum_acc = df_n[df_n['model_type'] == 'Quantum']['best_test_acc'].mean()
        classical_acc = df_n[df_n['model_type'] == 'Classical']['best_test_acc'].mean()

        advantages[n_bits] = {
            'quantum_acc': quantum_acc,
            'classical_acc': classical_acc,
            'advantage': quantum_acc - classical_acc
        }

    advantages_df = pd.DataFrame(advantages).T
    print("\nAdvantage at each complexity level:")
    print(advantages_df)

    # Check if advantage increases with complexity
    if len(advantages) >= 2:
        n_bits_sorted = sorted(advantages.keys())
        advantages_sorted = [advantages[n]['advantage'] for n in n_bits_sorted]

        if all(advantages_sorted[i] <= advantages_sorted[i+1] for i in range(len(advantages_sorted)-1)):
            print("\n  ✓ GOLD STANDARD MET: Quantum advantage increases with problem complexity!")
        else:
            print("\n  ✗ Gold Standard not met: Advantage does not consistently increase")
    else:
        print("\n  Insufficient data to assess Gold Standard (need multiple n_bits values)")

    return advantages_df


def create_plots(df, output_dir="experiments/forrelation_results"):
    """Create visualization plots."""
    output_path = Path(output_dir)

    # Categorize models
    df['model_type'] = df['model_name'].apply(
        lambda x: 'Quantum' if 'quantum' in x else 'Classical'
    )

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Plot 1: Accuracy vs Sequence Length (Sample Efficiency)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for n_bits, ax in zip([6, 8], axes):
        df_n = df[df['n_bits'] == n_bits]

        if len(df_n) == 0:
            continue

        # Group and plot
        for model_type in ['Quantum', 'Classical']:
            data = df_n[df_n['model_type'] == model_type].groupby('seq_len').agg({
                'best_test_acc': ['mean', 'std']
            })

            x = data.index
            y_mean = data[('best_test_acc', 'mean')]
            y_std = data[('best_test_acc', 'std')]

            ax.plot(x, y_mean, marker='o', label=model_type, linewidth=2)
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

        ax.set_xlabel('Sequence Length (L)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'Sample Efficiency (n_bits={n_bits})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])

    plt.tight_layout()
    plot_file = output_path / "forrelation_sample_efficiency.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {plot_file}")
    plt.close()

    # Plot 2: Model Comparison Heatmap
    pivot_table = df.pivot_table(
        values='best_test_acc',
        index='model_name',
        columns='seq_len',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title('Model Performance Heatmap (Accuracy)', fontsize=14)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.tight_layout()

    plot_file = output_path / "forrelation_heatmap.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    plt.close()


def main():
    print("="*80)
    print("FORRELATION RESULTS AGGREGATION AND ANALYSIS")
    print("="*80)

    # Load results
    df = load_all_results()

    if len(df) == 0:
        print("\nNo results found! Please run experiments first.")
        return

    print(f"\nLoaded {len(df)} results")
    print(f"Models: {df['model_name'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"n_bits: {sorted(df['n_bits'].unique())}")
    print(f"Sequence lengths: {sorted(df['seq_len'].unique())}")

    # Save full results to CSV
    output_file = Path("experiments/forrelation_results/forrelation_all_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\nSaved complete results to: {output_file}")

    # Phase 3: Sample Efficiency Analysis (Silver Standard)
    if 6 in df['n_bits'].unique():
        sample_efficiency = analyze_sample_efficiency(df, n_bits=6)

    # Phase 4: Scaling Analysis (Gold Standard)
    if len(df['n_bits'].unique()) >= 2:
        scaling = analyze_scaling(df)

    # Create plots
    create_plots(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - forrelation_all_results.csv")
    print("  - forrelation_sample_efficiency.png")
    print("  - forrelation_heatmap.png")
    print("="*80)


if __name__ == "__main__":
    main()
