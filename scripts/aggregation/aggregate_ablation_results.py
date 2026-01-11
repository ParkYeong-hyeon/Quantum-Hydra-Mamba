#!/usr/bin/env python3
"""
Aggregate Results from Ablation Study on PhysioNet EEG
Computes statistics for ICML/NeurIPS paper (from ABLATION_STUDY_PLAN_V3.md Section 7.6)

Outputs:
  - Summary tables (mean ± std across seeds)
  - Pairwise comparisons (paired t-test, Cohen's d, 95% CI)
  - Cross-frequency analysis (sequence length scaling)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import argparse


# ============================================
# Model Registry (same as run_ablation_eeg.py)
# ============================================
MODEL_REGISTRY = {
    '1a': {'name': 'QuantumTransformer', 'group': 1, 'feat': 'Q', 'mix': 'C', 'type': 'Trans'},
    '1b': {'name': 'QuantumMambaSSM', 'group': 1, 'feat': 'Q', 'mix': 'C', 'type': 'Mamba'},
    '1c': {'name': 'QuantumHydraSSM', 'group': 1, 'feat': 'Q', 'mix': 'C', 'type': 'Hydra'},
    '2a': {'name': 'ClassicalQuantumAttention', 'group': 2, 'feat': 'C', 'mix': 'Q', 'type': 'Trans'},
    '2b': {'name': 'ClassicalMambaQuantumSSM', 'group': 2, 'feat': 'C', 'mix': 'Q', 'type': 'Mamba'},
    '2c': {'name': 'ClassicalHydraQuantumSSM', 'group': 2, 'feat': 'C', 'mix': 'Q', 'type': 'Hydra'},
    '3a': {'name': 'ClassicalTransformer', 'group': 3, 'feat': 'C', 'mix': 'C', 'type': 'Trans'},
    '3b': {'name': 'TrueClassicalMamba', 'group': 3, 'feat': 'C', 'mix': 'C', 'type': 'Mamba'},
    '3c': {'name': 'TrueClassicalHydra', 'group': 3, 'feat': 'C', 'mix': 'C', 'type': 'Hydra'},
    '4a': {'name': 'QuantumTransformerE2E', 'group': 4, 'feat': 'Q', 'mix': 'Q', 'type': 'Trans'},
    '4b': {'name': 'QuantumMambaE2E', 'group': 4, 'feat': 'Q', 'mix': 'Q', 'type': 'Mamba'},
    '4c': {'name': 'QuantumHydraE2E', 'group': 4, 'feat': 'Q', 'mix': 'Q', 'type': 'Hydra'},
}

SAMPLING_FREQS = [40, 80, 160]
SEEDS = [2024, 2025, 2026, 2027, 2028]


def load_results(input_dir):
    """Load all result JSON files from input directory."""
    input_path = Path(input_dir)
    results = []

    # Find all result files
    for json_file in input_path.glob("ablation_*_results.json"):
        with open(json_file, 'r') as f:
            result = json.load(f)
            results.append(result)

    print(f"Loaded {len(results)} result files from {input_dir}")
    return results


def organize_results(results):
    """Organize results by (model_id, sampling_freq) → list of results across seeds."""
    organized = defaultdict(list)

    for r in results:
        model_id = r['model_id']
        freq = r['hyperparameters']['sampling_freq']
        key = (model_id, freq)
        organized[key].append(r)

    return organized


def compute_summary_stats(organized):
    """Compute mean ± std for each (model, freq) combination."""
    summary = []

    for (model_id, freq), result_list in organized.items():
        if len(result_list) == 0:
            continue

        model_info = MODEL_REGISTRY[model_id]

        accs = [r['test_acc'] for r in result_list]
        aucs = [r['test_auc'] for r in result_list]
        f1s = [r['test_f1'] for r in result_list]
        times = [r['training_time'] for r in result_list]

        summary.append({
            'model_id': model_id,
            'model_name': model_info['name'],
            'group': model_info['group'],
            'feat': model_info['feat'],
            'mix': model_info['mix'],
            'type': model_info['type'],
            'freq': freq,
            'n_seeds': len(result_list),
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'acc_values': accs,  # Keep raw values for statistical tests
            'auc_values': aucs,
        })

    return pd.DataFrame(summary)


def compute_pairwise_comparison(results_a, results_b, alpha=0.05, n_comparisons=66):
    """
    Compute pairwise statistics between two models.
    From ABLATION_STUDY_PLAN_V3.md Section 7.6.1
    """
    if len(results_a) != len(results_b) or len(results_a) < 2:
        return None

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results_a, results_b)

    # Effect size (Cohen's d for paired samples)
    diff = np.array(results_a) - np.array(results_b)
    if diff.std() > 0:
        cohens_d = diff.mean() / diff.std()
    else:
        cohens_d = 0.0

    # 95% Confidence Interval
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))

    # Bonferroni correction
    bonferroni_alpha = alpha / n_comparisons

    return {
        'mean_a': np.mean(results_a),
        'std_a': np.std(results_a),
        'mean_b': np.mean(results_b),
        'std_b': np.std(results_b),
        'mean_diff': diff.mean(),
        'p_value': p_value,
        'significant': p_value < alpha,
        'significant_bonferroni': p_value < bonferroni_alpha,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'ci_95_low': ci[0],
        'ci_95_high': ci[1],
    }


def analyze_q1_where_quantum(summary_df, organized):
    """
    Q1: Where should quantum be applied?
    Compare Groups 1 vs 2 vs 3 vs 4 for each mixing type
    """
    print("\n" + "=" * 80)
    print("Q1: WHERE SHOULD QUANTUM BE APPLIED?")
    print("=" * 80)

    comparisons = []

    for freq in SAMPLING_FREQS:
        print(f"\n--- {freq} Hz ---")

        for mix_type in ['Trans', 'Mamba', 'Hydra']:
            # Get models of this type across all groups
            type_models = {
                f'{g}{t}': m for m, info in MODEL_REGISTRY.items()
                for g, t in [(info['group'], 'a' if info['type'] == 'Trans' else 'b' if info['type'] == 'Mamba' else 'c')]
                if info['type'] == mix_type
            }

            # Get Group 1 vs Group 2 comparison (Q-Feat vs Q-Mix)
            model_1 = [m for m, info in MODEL_REGISTRY.items() if info['group'] == 1 and info['type'] == mix_type][0]
            model_2 = [m for m, info in MODEL_REGISTRY.items() if info['group'] == 2 and info['type'] == mix_type][0]

            if (model_1, freq) in organized and (model_2, freq) in organized:
                acc_1 = [r['test_acc'] for r in organized[(model_1, freq)]]
                acc_2 = [r['test_acc'] for r in organized[(model_2, freq)]]

                if len(acc_1) >= 2 and len(acc_2) >= 2:
                    comp = compute_pairwise_comparison(acc_2, acc_1)  # Group 2 - Group 1
                    if comp:
                        sig = "*" if comp['significant'] else ""
                        print(f"  {mix_type}: Group2 vs Group1 = {comp['mean_diff']:+.4f}{sig} "
                              f"(d={comp['cohens_d']:.2f}, p={comp['p_value']:.4f})")

                        comparisons.append({
                            'freq': freq,
                            'mix_type': mix_type,
                            'comparison': 'G2 vs G1',
                            **comp
                        })

    return pd.DataFrame(comparisons)


def analyze_q2_which_mixing(summary_df, organized):
    """
    Q2: Which mixing mechanism benefits most from quantum?
    Compare 2a vs 2b vs 2c
    """
    print("\n" + "=" * 80)
    print("Q2: WHICH MIXING BENEFITS MOST FROM QUANTUM?")
    print("=" * 80)

    comparisons = []

    for freq in SAMPLING_FREQS:
        print(f"\n--- {freq} Hz (Group 2: Classical Features → Quantum Mixing) ---")

        models_g2 = ['2a', '2b', '2c']
        results_g2 = {}

        for m in models_g2:
            if (m, freq) in organized:
                results_g2[m] = [r['test_acc'] for r in organized[(m, freq)]]

        if len(results_g2) == 3:
            # Print accuracies
            for m in models_g2:
                mean_acc = np.mean(results_g2[m])
                std_acc = np.std(results_g2[m])
                print(f"  {m} ({MODEL_REGISTRY[m]['type']}): {mean_acc:.4f} ± {std_acc:.4f}")

            # Pairwise comparisons
            pairs = [('2a', '2b'), ('2a', '2c'), ('2b', '2c')]
            for m1, m2 in pairs:
                comp = compute_pairwise_comparison(results_g2[m1], results_g2[m2])
                if comp:
                    sig = "*" if comp['significant'] else ""
                    print(f"    {m1} vs {m2}: Δ={comp['mean_diff']:+.4f}{sig} (p={comp['p_value']:.4f})")

                    comparisons.append({
                        'freq': freq,
                        'comparison': f'{m1} vs {m2}',
                        **comp
                    })

    return pd.DataFrame(comparisons)


def analyze_q3_e2e_quantum(summary_df, organized):
    """
    Q3: Does end-to-end quantum provide synergistic benefits?
    Compare Group 3 (baseline) vs Group 4 (E2E quantum)
    """
    print("\n" + "=" * 80)
    print("Q3: DOES END-TO-END QUANTUM HELP?")
    print("=" * 80)

    comparisons = []

    for freq in SAMPLING_FREQS:
        print(f"\n--- {freq} Hz (Group 4 vs Group 3) ---")

        for mix_type in ['Trans', 'Mamba', 'Hydra']:
            model_3 = [m for m, info in MODEL_REGISTRY.items() if info['group'] == 3 and info['type'] == mix_type][0]
            model_4 = [m for m, info in MODEL_REGISTRY.items() if info['group'] == 4 and info['type'] == mix_type][0]

            if (model_3, freq) in organized and (model_4, freq) in organized:
                acc_3 = [r['test_acc'] for r in organized[(model_3, freq)]]
                acc_4 = [r['test_acc'] for r in organized[(model_4, freq)]]

                if len(acc_3) >= 2 and len(acc_4) >= 2:
                    comp = compute_pairwise_comparison(acc_4, acc_3)  # Group 4 - Group 3
                    if comp:
                        sig = "*" if comp['significant'] else ""
                        print(f"  {mix_type}: G4 vs G3 = {comp['mean_diff']:+.4f}{sig} "
                              f"(d={comp['cohens_d']:.2f}, p={comp['p_value']:.4f})")

                        comparisons.append({
                            'freq': freq,
                            'mix_type': mix_type,
                            'comparison': 'G4 vs G3',
                            **comp
                        })

    return pd.DataFrame(comparisons)


def analyze_sequence_length_scaling(summary_df, organized):
    """
    Analyze how models scale with sequence length (40 → 80 → 160 Hz)
    """
    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH SCALING ANALYSIS")
    print("=" * 80)

    scaling_results = []

    for model_id in MODEL_REGISTRY.keys():
        accs_by_freq = {}
        for freq in SAMPLING_FREQS:
            if (model_id, freq) in organized:
                accs_by_freq[freq] = np.mean([r['test_acc'] for r in organized[(model_id, freq)]])

        if len(accs_by_freq) == 3:
            # Compute scaling (160 Hz - 40 Hz)
            delta = accs_by_freq[160] - accs_by_freq[40]
            scales_well = delta > -0.02  # Less than 2% drop

            scaling_results.append({
                'model_id': model_id,
                'group': MODEL_REGISTRY[model_id]['group'],
                'type': MODEL_REGISTRY[model_id]['type'],
                'acc_40': accs_by_freq[40],
                'acc_80': accs_by_freq[80],
                'acc_160': accs_by_freq[160],
                'delta_160_40': delta,
                'scales_well': scales_well,
            })

    df_scaling = pd.DataFrame(scaling_results)

    if not df_scaling.empty:
        print("\n" + "-" * 60)
        print(f"{'Model':<6} {'Type':<8} {'40Hz':>8} {'80Hz':>8} {'160Hz':>8} {'Δ':>8} {'Scales?'}")
        print("-" * 60)

        for _, row in df_scaling.sort_values(['group', 'type']).iterrows():
            scales = "✓" if row['scales_well'] else "✗"
            print(f"{row['model_id']:<6} {row['type']:<8} "
                  f"{row['acc_40']:>8.4f} {row['acc_80']:>8.4f} {row['acc_160']:>8.4f} "
                  f"{row['delta_160_40']:>+8.4f} {scales:>7}")

    return df_scaling


def create_main_results_table(summary_df):
    """Create Table 1: Main Results (per model × frequency)."""
    print("\n" + "=" * 80)
    print("TABLE 1: MAIN RESULTS")
    print("=" * 80)

    # Pivot table: rows = models, columns = frequencies
    pivot = summary_df.pivot_table(
        index=['model_id', 'group', 'type'],
        columns='freq',
        values='acc_mean',
        aggfunc='first'
    ).reset_index()

    print("\n" + "-" * 70)
    print(f"{'Model':<6} {'Grp':<4} {'Type':<8} {'40Hz':>12} {'80Hz':>12} {'160Hz':>12}")
    print("-" * 70)

    for _, row in pivot.sort_values(['group', 'type']).iterrows():
        print(f"{row['model_id']:<6} {row['group']:<4} {row['type']:<8} "
              f"{row.get(40, np.nan):>12.4f} {row.get(80, np.nan):>12.4f} {row.get(160, np.nan):>12.4f}")

    return pivot


def save_results(summary_df, q1_df, q2_df, q3_df, scaling_df, output_dir):
    """Save all results to CSV files."""
    output_path = Path(output_dir)

    # Summary
    summary_df.to_csv(output_path / "ablation_summary.csv", index=False)

    # Statistical comparisons
    if not q1_df.empty:
        q1_df.to_csv(output_path / "ablation_q1_comparisons.csv", index=False)
    if not q2_df.empty:
        q2_df.to_csv(output_path / "ablation_q2_comparisons.csv", index=False)
    if not q3_df.empty:
        q3_df.to_csv(output_path / "ablation_q3_comparisons.csv", index=False)
    if not scaling_df.empty:
        scaling_df.to_csv(output_path / "ablation_scaling.csv", index=False)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Ablation Study Results")
    parser.add_argument("--input-dir", type=str, default="./results/ablation_eeg",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    # Load results
    results = load_results(args.input_dir)

    if len(results) == 0:
        print("No results found. Please run experiments first.")
        return

    # Organize by (model, freq)
    organized = organize_results(results)

    # Compute summary statistics
    summary_df = compute_summary_stats(organized)

    # Create main results table
    create_main_results_table(summary_df)

    # Analyze research questions
    q1_df = analyze_q1_where_quantum(summary_df, organized)
    q2_df = analyze_q2_which_mixing(summary_df, organized)
    q3_df = analyze_q3_e2e_quantum(summary_df, organized)

    # Sequence length scaling
    scaling_df = analyze_sequence_length_scaling(summary_df, organized)

    # Save results
    save_results(summary_df, q1_df, q2_df, q3_df, scaling_df, output_dir)

    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Total experiments analyzed: {len(results)}")
    print(f"Models: {summary_df['model_id'].nunique()}")
    print(f"Frequencies: {summary_df['freq'].nunique()}")
    print(f"Average seeds per condition: {summary_df['n_seeds'].mean():.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
