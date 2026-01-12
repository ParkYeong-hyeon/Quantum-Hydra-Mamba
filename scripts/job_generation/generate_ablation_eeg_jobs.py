#!/usr/bin/env python3
"""
Generate SLURM job scripts for the Ablation Study on PhysioNet EEG
12 models × 3 sampling frequencies × 5 seeds = 180 jobs

Based on ABLATION_STUDY_PLAN_V3.md Section 7.3
"""

from pathlib import Path
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    PROJECT_ROOT,
    PY,
    get_output_dir,
    get_jobs_dir,
    get_script_path,
    get_slurm_config,
)

# ============================================
# Configuration
# ============================================

# 14 Models from 2×2×3 Factorial Design + E2E Superposition
MODEL_IDS = [
    # Group 1: Quantum Features → Classical Mixing
    '1a', '1b', '1c',
    # Group 2: Classical Features → Quantum Mixing
    '2a', '2b', '2c', '2d', '2e',
    # Group 3: Classical Features → Classical Mixing (Baseline)
    '3a', '3b', '3c',
    # Group 4: Quantum Features → Quantum Mixing (E2E)
    '4a', '4b', '4c', '4d', '4e',
]

MODEL_NAMES = {
    '1a': 'QuantumTransformer',
    '1b': 'QuantumMambaSSM',
    '1c': 'QuantumHydraSSM',
    '2a': 'ClassicalQuantumAttention',
    '2b': 'ClassicalMambaQuantumSSM',
    '2c': 'ClassicalHydraQuantumSSM',
    '2d': 'QuantumMambaHydraSSM',
    '2e': 'QuantumHydraHydraSSM',
    '3a': 'ClassicalTransformer',
    '3b': 'TrueClassicalMamba',
    '3c': 'TrueClassicalHydra',
    '4a': 'QuantumTransformerE2E',
    '4b': 'QuantumMambaE2E',
    '4c': 'QuantumHydraE2E',
    '4d': 'QuantumMambaE2ESuperposition',
    '4e': 'QuantumHydraE2ESuperposition',
}

# 3 Sampling Frequencies (Sequence Length Conditions)
SAMPLING_FREQS = [40, 80, 160]  # Hz → ~124, 248, 496 timesteps

# 3 Seeds for Statistical Rigor
SEEDS = [2024, 2025, 2026]

# Hyperparameters (from ABLATION_STUDY_PLAN_V3.md Section 7.4)
N_QUBITS = 6
N_LAYERS = 2
D_MODEL = 128
D_STATE = 16
N_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING = 10
SAMPLE_SIZE = 109  # All 109 subjects

# Paths (from config)
OUTPUT_DIR = get_output_dir('ablation_eeg')
JOBS_DIR = get_jobs_dir('ablation_eeg')
SCRIPT_PATH = get_script_path('run_ablation_eeg.py')

# SLURM Configuration (from config, can be overridden per experiment)
slurm_config = get_slurm_config()
SLURM_ACCOUNT = slurm_config['account']
SLURM_CONSTRAINT = slurm_config['constraint']
SLURM_QOS = slurm_config['qos']
SLURM_TIME = slurm_config['time']
SLURM_NODES = slurm_config['nodes']
SLURM_GPUS = slurm_config['gpus']
SLURM_CPUS = slurm_config['cpus_per_task']


def create_job_script(model_id, sampling_freq, seed, job_num, total_jobs):
    """Create a single SLURM job script."""

    model_name = MODEL_NAMES[model_id]
    job_name = f"abl_{model_id}_{sampling_freq}Hz_s{seed}"
    log_file = str(OUTPUT_DIR / "logs" / f"{job_name}.log")

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --constraint={SLURM_CONSTRAINT}
#SBATCH --qos={SLURM_QOS}
#SBATCH -t {SLURM_TIME}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --gpus={SLURM_GPUS}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --output={log_file}
#SBATCH --error={log_file}

# ============================================
# Ablation Study Job [{job_num}/{total_jobs}]
# ============================================
# Model: {model_id} ({model_name})
# Sampling Freq: {sampling_freq} Hz
# Seed: {seed}
# ============================================

echo "============================================"
echo "Ablation Study - EEG Classification"
echo "============================================"
echo "Job: {job_name}"
echo "Model: {model_id} ({model_name})"
echo "Sampling Freq: {sampling_freq} Hz"
echo "Seed: {seed}"
echo "Started: $(date)"
echo "============================================"

# Use Python directly from environment
PYTHON_CMD="{PY}"

# Navigate to project root
cd {str(PROJECT_ROOT)}

# Run training (with --resume to automatically continue from checkpoint if available)
$PYTHON_CMD {str(SCRIPT_PATH)} \\
    --model-id {model_id} \\
    --n-qubits {N_QUBITS} \\
    --n-layers {N_LAYERS} \\
    --d-model {D_MODEL} \\
    --d-state {D_STATE} \\
    --n-epochs {N_EPOCHS} \\
    --batch-size {BATCH_SIZE} \\
    --lr {LEARNING_RATE} \\
    --weight-decay {WEIGHT_DECAY} \\
    --early-stopping {EARLY_STOPPING} \\
    --sample-size {SAMPLE_SIZE} \\
    --sampling-freq {sampling_freq} \\
    --seed {seed} \\
    --output-dir {str(OUTPUT_DIR)} \\
    --device cuda \\
    --resume

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
"""

    return script


def main():
    """Generate all 180 job scripts."""

    # Calculate total jobs
    total_jobs = len(MODEL_IDS) * len(SAMPLING_FREQS) * len(SEEDS)

    print("=" * 80)
    print("Generating Ablation Study Job Scripts for PhysioNet EEG")
    print("=" * 80)
    print(f"Models: {len(MODEL_IDS)}")
    print(f"Sampling Frequencies: {SAMPLING_FREQS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total Jobs: {total_jobs}")
    print("=" * 80)

    # Create directories
    jobs_path = JOBS_DIR
    jobs_path.mkdir(parents=True, exist_ok=True)

    logs_path = OUTPUT_DIR / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    # Generate job scripts
    job_files = []
    job_num = 1

    for sampling_freq in SAMPLING_FREQS:
        freq_dir = jobs_path / f"{sampling_freq}Hz"
        freq_dir.mkdir(exist_ok=True)

        for model_id in MODEL_IDS:
            for seed in SEEDS:
                # Create job script
                script = create_job_script(model_id, sampling_freq, seed, job_num, total_jobs)

                # Save to file
                job_filename = f"abl_{model_id}_{sampling_freq}Hz_s{seed}.sh"
                job_path = freq_dir / job_filename

                with open(job_path, 'w') as f:
                    f.write(script)

                # Make executable
                job_path.chmod(0o755)

                job_files.append(str(job_path))

                if job_num % 20 == 0 or job_num == total_jobs:
                    print(f"  [{job_num:3d}/{total_jobs}] Created jobs...")

                job_num += 1

    print(f"\nGenerated {len(job_files)} job scripts in {jobs_path}")

    # ============================================
    # Create submission scripts
    # ============================================

    # 1. Submit all jobs
    submit_all = f"""#!/bin/bash
# Submit ALL {total_jobs} ablation study jobs
# WARNING: This will submit many jobs at once!

echo "Submitting {total_jobs} ablation study jobs..."
echo "Press Ctrl+C to cancel..."
sleep 3

"""
    for freq in SAMPLING_FREQS:
        submit_all += f"\n# ===== {freq} Hz Jobs =====\n"
        for model_id in MODEL_IDS:
            for seed in SEEDS:
                job_file = str(JOBS_DIR / f"{freq}Hz" / f"abl_{model_id}_{freq}Hz_s{seed}.sh")
                submit_all += f"sbatch {job_file}\n"
        submit_all += "sleep 1  # Pause between frequency batches\n"

    submit_all += f"""
echo "============================================"
echo "All {total_jobs} jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Logs in: {OUTPUT_DIR}/logs/"
echo "============================================"
"""

    submit_all_path = jobs_path / "submit_all.sh"
    with open(submit_all_path, 'w') as f:
        f.write(submit_all)
    submit_all_path.chmod(0o755)

    # 2. Submit by frequency
    for freq in SAMPLING_FREQS:
        n_freq_jobs = len(MODEL_IDS) * len(SEEDS)
        submit_freq = f"""#!/bin/bash
# Submit all {n_freq_jobs} jobs for {freq} Hz condition
echo "Submitting {n_freq_jobs} jobs for {freq} Hz..."

"""
        for model_id in MODEL_IDS:
            for seed in SEEDS:
                job_file = str(JOBS_DIR / f"{freq}Hz" / f"abl_{model_id}_{freq}Hz_s{seed}.sh")
                submit_freq += f"sbatch {job_file}\n"

        submit_freq += f"""
echo "Submitted {n_freq_jobs} jobs for {freq} Hz"
"""

        submit_freq_path = jobs_path / f"submit_{freq}Hz.sh"
        with open(submit_freq_path, 'w') as f:
            f.write(submit_freq)
        submit_freq_path.chmod(0o755)

    # 3. Submit by model
    for model_id in MODEL_IDS:
        n_model_jobs = len(SAMPLING_FREQS) * len(SEEDS)
        submit_model = f"""#!/bin/bash
# Submit all {n_model_jobs} jobs for model {model_id} ({MODEL_NAMES[model_id]})
echo "Submitting {n_model_jobs} jobs for {model_id}..."

"""
        for freq in SAMPLING_FREQS:
            for seed in SEEDS:
                job_file = str(JOBS_DIR / f"{freq}Hz" / f"abl_{model_id}_{freq}Hz_s{seed}.sh")
                submit_model += f"sbatch {job_file}\n"

        submit_model += f"""
echo "Submitted {n_model_jobs} jobs for model {model_id}"
"""

        submit_model_path = jobs_path / f"submit_model_{model_id}.sh"
        with open(submit_model_path, 'w') as f:
            f.write(submit_model)
        submit_model_path.chmod(0o755)

    # 4. Submit by group
    groups = {
        'group1': ['1a', '1b', '1c'],
        'group2': ['2a', '2b', '2c'],
        'group3': ['3a', '3b', '3c'],
        'group4': ['4a', '4b', '4c'],
    }

    for group_name, group_models in groups.items():
        n_group_jobs = len(group_models) * len(SAMPLING_FREQS) * len(SEEDS)
        submit_group = f"""#!/bin/bash
# Submit all {n_group_jobs} jobs for {group_name}
# Models: {', '.join(group_models)}
echo "Submitting {n_group_jobs} jobs for {group_name}..."

"""
        for model_id in group_models:
            for freq in SAMPLING_FREQS:
                for seed in SEEDS:
                    job_file = str(JOBS_DIR / f"{freq}Hz" / f"abl_{model_id}_{freq}Hz_s{seed}.sh")
                    submit_group += f"sbatch {job_file}\n"

        submit_group += f"""
echo "Submitted {n_group_jobs} jobs for {group_name}"
"""

        submit_group_path = jobs_path / f"submit_{group_name}.sh"
        with open(submit_group_path, 'w') as f:
            f.write(submit_group)
        submit_group_path.chmod(0o755)

    # ============================================
    # Print summary
    # ============================================

    print("\n" + "=" * 80)
    print("Job Generation Complete!")
    print("=" * 80)

    print(f"\n📁 Job Scripts Location: {jobs_path}")
    print(f"   ├── 40Hz/   ({len(MODEL_IDS) * len(SEEDS)} jobs)")
    print(f"   ├── 80Hz/   ({len(MODEL_IDS) * len(SEEDS)} jobs)")
    print(f"   └── 160Hz/  ({len(MODEL_IDS) * len(SEEDS)} jobs)")

    print(f"\n📋 Submission Scripts:")
    print(f"   • submit_all.sh          - Submit all {total_jobs} jobs")
    print(f"   • submit_40Hz.sh         - Submit 40 Hz jobs only (60 jobs)")
    print(f"   • submit_80Hz.sh         - Submit 80 Hz jobs only (60 jobs)")
    print(f"   • submit_160Hz.sh        - Submit 160 Hz jobs only (60 jobs)")
    print(f"   • submit_group1.sh       - Submit Group 1 (Q-Feat → C-Mix, 45 jobs)")
    print(f"   • submit_group2.sh       - Submit Group 2 (C-Feat → Q-Mix, 45 jobs)")
    print(f"   • submit_group3.sh       - Submit Group 3 (Baseline, 45 jobs)")
    print(f"   • submit_group4.sh       - Submit Group 4 (E2E, 45 jobs)")
    print(f"   • submit_model_*.sh      - Submit by model (15 jobs each)")

    print(f"\n🚀 Quick Start:")
    print(f"   # Submit all jobs:")
    print(f"   bash {jobs_path}/submit_all.sh")
    print(f"")
    print(f"   # Submit just 80 Hz (recommended to start):")
    print(f"   bash {jobs_path}/submit_80Hz.sh")
    print(f"")
    print(f"   # Submit classical baselines first:")
    print(f"   bash {jobs_path}/submit_group3.sh")

    print(f"\n📊 Monitor Progress:")
    print(f"   squeue -u $USER")
    print(f"   tail -f {OUTPUT_DIR}/logs/*.log")

    print(f"\n📈 After Completion:")
    print(f"   python scripts/aggregation/aggregate_ablation_results.py --input-dir {OUTPUT_DIR}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
