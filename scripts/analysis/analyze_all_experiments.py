#!/usr/bin/env python3
"""
Comprehensive analysis of all quantum experiment logs
"""
import os
from pathlib import Path
from collections import defaultdict

# Define all experiments
datasets = ['dna', 'mnist', 'eeg']
models = ['quantum_hydra', 'quantum_hydra_hybrid', 'quantum_mamba', 'quantum_mamba_hybrid', 'classical_hydra', 'classical_mamba']
seeds = [2024, 2025, 2026, 2027, 2028]

# Results tracking
results = {
    'successful': [],
    'failed_with_error': [],
    'never_ran': []
}

error_types = defaultdict(list)

def check_log_for_success(log_path):
    """Check if log indicates successful completion"""
    if not os.path.exists(log_path):
        return 'no_log', None

    with open(log_path, 'r') as f:
        content = f.read()

    # Check for errors FIRST (higher priority)
    if 'IndexError' in content:
        # Extract error message
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'IndexError' in line:
                error_msg = line
                if i+1 < len(lines):
                    error_msg += '\n' + lines[i+1]
                return 'error', f"IndexError: {error_msg}"

    # Check for critical errors (Traceback usually indicates crash)
    if 'Traceback (most recent call last)' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Error:' in line or 'Exception:' in line:
                return 'error', line[:200]
        return 'error', 'Traceback found but error type unclear'

    # Check for success indicators (relaxed criteria)
    if 'TRAINING COMPLETE!' in content or 'Training Complete!' in content:
        if 'Results saved to' in content:
            # Check if training had issues (NaN loss)
            if 'Train Loss: nan' in content or 'Val Loss: nan' in content:
                return 'completed_with_nan', None
            return 'success', None
        else:
            return 'completed_no_save', None

    # Check if log is empty or very short
    if len(content) < 100:
        return 'empty_log', None

    # Log exists but unclear status
    return 'unclear', None

print("="*80)
print("COMPREHENSIVE QUANTUM EXPERIMENT STATUS ANALYSIS")
print("="*80)

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'='*80}\n")

    dataset_stats = {
        'successful': 0,
        'failed': 0,
        'never_ran': 0,
        'unclear': 0,
        'nan_issues': 0
    }

    for model in models:
        print(f"\n  Model: {model}")
        print(f"  {'-'*70}")

        for seed in seeds:
            exp_name = f"{dataset}_{model}_seed{seed}"

            # Check Python log
            python_log = f"./experiments/{dataset}_results/logs/{exp_name}.log"
            slurm_log = f"./experiments/{dataset}_results/logs/{exp_name}.sh.log"

            status, error = check_log_for_success(python_log)

            # If no Python log, check SLURM log
            if status == 'no_log':
                slurm_status, slurm_error = check_log_for_success(slurm_log)
                if slurm_status == 'no_log':
                    print(f"    ❌ {exp_name}: NEVER RAN (no logs)")
                    results['never_ran'].append(exp_name)
                    dataset_stats['never_ran'] += 1
                elif slurm_status == 'success':
                    print(f"    ✅ {exp_name}: SUCCESS (SLURM log)")
                    results['successful'].append(exp_name)
                    dataset_stats['successful'] += 1
                elif slurm_status == 'completed_with_nan':
                    print(f"    ⚠️  {exp_name}: COMPLETED WITH NaN (SLURM log)")
                    dataset_stats['nan_issues'] += 1
                elif slurm_status == 'error':
                    print(f"    ❌ {exp_name}: FAILED (SLURM log)")
                    print(f"         Error: {slurm_error[:100]}")
                    results['failed_with_error'].append((exp_name, slurm_error))
                    error_types[slurm_error[:50]].append(exp_name)
                    dataset_stats['failed'] += 1
                else:
                    print(f"    ⚠️  {exp_name}: UNCLEAR (SLURM log exists, status: {slurm_status})")
                    dataset_stats['unclear'] += 1
            elif status == 'success':
                print(f"    ✅ {exp_name}: SUCCESS")
                results['successful'].append(exp_name)
                dataset_stats['successful'] += 1
            elif status == 'completed_with_nan':
                print(f"    ⚠️  {exp_name}: COMPLETED (NaN loss issues)")
                dataset_stats['nan_issues'] += 1
            elif status == 'completed_no_save':
                print(f"    ⚠️  {exp_name}: COMPLETED (no results saved)")
                dataset_stats['unclear'] += 1
            elif status == 'error':
                print(f"    ❌ {exp_name}: FAILED")
                print(f"         Error: {error[:100]}")
                results['failed_with_error'].append((exp_name, error))
                error_types[error[:50]].append(exp_name)
                dataset_stats['failed'] += 1
            elif status == 'empty_log':
                print(f"    ⚠️  {exp_name}: EMPTY LOG")
                dataset_stats['unclear'] += 1
            else:
                print(f"    ⚠️  {exp_name}: UNCLEAR (log exists but no clear success/failure)")
                dataset_stats['unclear'] += 1

    # Dataset summary
    print(f"\n  {'='*70}")
    print(f"  {dataset.upper()} SUMMARY:")
    print(f"    ✅ Successful: {dataset_stats['successful']}/30")
    print(f"    ❌ Failed with crash: {dataset_stats['failed']}/30")
    print(f"    ⚠️  Completed with NaN: {dataset_stats['nan_issues']}/30")
    print(f"    ⚠️  Never ran: {dataset_stats['never_ran']}/30")
    print(f"    ⚠️  Unclear: {dataset_stats['unclear']}/30")
    print(f"  {'='*70}")

# Overall summary
print(f"\n\n{'='*80}")
print("OVERALL SUMMARY (ALL 90 EXPERIMENTS)")
print(f"{'='*80}")
print(f"✅ Successful: {len(results['successful'])}/90")
print(f"❌ Failed with error: {len(results['failed_with_error'])}/90")
print(f"⚠️  Never ran (no logs): {len(results['never_ran'])}/90")

# Error type breakdown
if error_types:
    print(f"\n{'='*80}")
    print("ERROR TYPE BREAKDOWN")
    print(f"{'='*80}")
    for error_type, exps in error_types.items():
        print(f"\n{error_type}")
        print(f"  Affected experiments ({len(exps)}):")
        for exp in exps:
            print(f"    - {exp}")

print(f"\n{'='*80}")
