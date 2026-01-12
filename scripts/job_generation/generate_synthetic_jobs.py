#!/usr/bin/env python3
"""
Generate SLURM job scripts for Synthetic Benchmark experiments.

Creates individual job scripts for each combination of:
  - Models: 1a, 1b, 1c, 2a, 2d, 2e, 3a, 3b, 3c, 4a, 4d, 4e (12 models)
  - Tasks: forrelation, adding_problem, selective_copy (3 tasks)
  - Sequence lengths: 100, 200, 500, 1000 (4 lengths)
  - Seeds: 2024, 2025, 2026 (3 seeds)

Total: 12 × 3 × 4 × 3 = 432 experiments

Usage:
    python generate_synthetic_jobs.py
    python generate_synthetic_jobs.py --output-dir ./slurm_jobs/synthetic
    python generate_synthetic_jobs.py --models 1c 2e 3b --tasks adding_problem selective_copy
"""

import os
import argparse
import sys
from pathlib import Path
from itertools import product

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

# Model configurations with estimated runtime (hours)
# All models use 24:00:00 as requested
MODEL_CONFIGS = {
    # Group 1: Quantum Features → Classical Mixing
    '1a': {'name': 'QuantumTransformer', 'time': '24:00:00', 'group': 1},
    '1b': {'name': 'QuantumMambaSSM', 'time': '24:00:00', 'group': 1},
    '1c': {'name': 'QuantumHydraSSM', 'time': '24:00:00', 'group': 1},

    # Group 2: Classical Features → Quantum Mixing
    '2a': {'name': 'ClassicalQuantumAttention', 'time': '24:00:00', 'group': 2},
    '2d': {'name': 'QuantumMambaHydraSSM', 'time': '24:00:00', 'group': 2},
    '2e': {'name': 'QuantumHydraHydraSSM', 'time': '24:00:00', 'group': 2},

    # Group 3: Classical Baselines
    '3a': {'name': 'ClassicalTransformer', 'time': '24:00:00', 'group': 3},
    '3b': {'name': 'TrueClassicalMamba', 'time': '24:00:00', 'group': 3},
    '3c': {'name': 'TrueClassicalHydra', 'time': '24:00:00', 'group': 3},

    # Group 4: E2E Quantum
    '4a': {'name': 'QuantumTransformerE2E', 'time': '24:00:00', 'group': 4},
    '4d': {'name': 'QuantumMambaE2E_Super', 'time': '24:00:00', 'group': 4},
    '4e': {'name': 'QuantumHydraE2E_Super', 'time': '24:00:00', 'group': 4},
}

# Task configurations
TASK_CONFIGS = {
    'forrelation': {'seq_lens': [50, 100, 200]},
    'adding_problem': {'seq_lens': [100, 200, 500, 1000]},
    'selective_copy': {'seq_lens': [100, 200, 500, 1000]},
}

# Default seeds
DEFAULT_SEEDS = [2024, 2025, 2026]

# SLURM Configuration (from config)
slurm_config = get_slurm_config()
SLURM_ACCOUNT = slurm_config['account']
SLURM_CONSTRAINT = slurm_config['constraint']
SLURM_QOS = slurm_config['qos']
SLURM_NODES = slurm_config['nodes']
SLURM_GPUS = slurm_config['gpus']
SLURM_CPUS = slurm_config['cpus_per_task']

# SLURM template
SLURM_TEMPLATE = f'''#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --constraint={SLURM_CONSTRAINT}
#SBATCH --qos={SLURM_QOS}
#SBATCH --time={{time}}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --gpus={SLURM_GPUS}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --output={{output_dir}}/logs/{{job_name}}_%j.out
#SBATCH --error={{output_dir}}/logs/{{job_name}}_%j.err

# ============================================
# Synthetic Benchmark Experiment
# ============================================
# Model: {{model_id}} ({{model_name}})
# Task: {{task}}
# Sequence Length: {{seq_len}}
# Seed: {{seed}}
# ============================================

echo "Starting job: {{job_name}}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Use Python directly from environment
PYTHON_CMD="{PY}"

# Navigate to project directory
cd {str(PROJECT_ROOT)}

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export PENNYLANE_DEVICE=default.qubit

# Run experiment
$PYTHON_CMD {str(get_script_path('run_synthetic_benchmark.py'))} \\
    --model-id {model_id} \\
    --task {task} \\
    --seq-len {seq_len} \\
    --seed {seed} \\
    --n-qubits 6 \\
    --n-layers 2 \\
    --d-model 128 \\
    --d-state 16 \\
    --n-epochs 100 \\
    --batch-size 32 \\
    --lr 0.001 \\
    --weight-decay 0.0001 \\
    --early-stopping 20 \\
    --num-markers 8 \\
    --output-dir ./results/synthetic_benchmarks \\
    --data-dir ./data/synthetic_benchmarks \\
    --device cuda

echo "Job completed: {job_name}"
echo "End time: $(date)"
'''

# Array job template (for running multiple experiments in one submission)
ARRAY_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=synthetic_{task}_batch
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time={max_time}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-{max_index}%{concurrent}
#SBATCH --output={output_dir}/logs/synthetic_{task}_%A_%a.out
#SBATCH --error={output_dir}/logs/synthetic_{task}_%A_%a.err

# ============================================
# Synthetic Benchmark Array Job
# Task: {task}
# Total experiments: {total}
# ============================================

echo "Array job started"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"

# Use Python directly from environment
PYTHON_CMD="{PY}"

cd {str(PROJECT_ROOT)}

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export PENNYLANE_DEVICE=default.qubit

# Define experiment configurations
MODELS=({models})
SEQ_LENS=({seq_lens})
SEEDS=({seeds})

# Calculate indices
n_models=${{#MODELS[@]}}
n_lens=${{#SEQ_LENS[@]}}
n_seeds=${{#SEEDS[@]}}

model_idx=$(( $SLURM_ARRAY_TASK_ID / (n_lens * n_seeds) ))
remainder=$(( $SLURM_ARRAY_TASK_ID % (n_lens * n_seeds) ))
len_idx=$(( remainder / n_seeds ))
seed_idx=$(( remainder % n_seeds ))

MODEL=${{MODELS[$model_idx]}}
SEQ_LEN=${{SEQ_LENS[$len_idx]}}
SEED=${{SEEDS[$seed_idx]}}

echo "Running: Model=$MODEL, Task={task}, SeqLen=$SEQ_LEN, Seed=$SEED"

$PYTHON_CMD {str(get_script_path('run_synthetic_benchmark.py'))} \\
    --model-id $MODEL \\
    --task {task} \\
    --seq-len $SEQ_LEN \\
    --seed $SEED \\
    --n-qubits 6 \\
    --n-layers 2 \\
    --d-model 128 \\
    --d-state 16 \\
    --n-epochs 100 \\
    --batch-size 32 \\
    --lr 0.001 \\
    --weight-decay 0.0001 \\
    --early-stopping 20 \\
    --num-markers 8 \\
    --output-dir ./results/synthetic_benchmarks \\
    --data-dir ./data/synthetic_benchmarks \\
    --device cuda

echo "Experiment completed"
echo "End time: $(date)"
'''


def generate_individual_jobs(models, tasks, seeds, output_dir):
    """Generate individual SLURM job scripts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    job_files = []
    total_jobs = 0

    for task in tasks:
        task_config = TASK_CONFIGS[task]
        seq_lens = task_config['seq_lens']

        for model_id in models:
            model_config = MODEL_CONFIGS[model_id]

            for seq_len in seq_lens:
                for seed in seeds:
                    job_name = f"syn_{model_id}_{task}_L{seq_len}_s{seed}"

                    script_content = SLURM_TEMPLATE.format(
                        job_name=job_name,
                        model_id=model_id,
                        model_name=model_config['name'],
                        task=task,
                        seq_len=seq_len,
                        seed=seed,
                        time=model_config['time'],
                        output_dir=str(output_dir)
                    )

                    script_path = output_dir / f"{job_name}.sh"
                    with open(script_path, 'w') as f:
                        f.write(script_content)

                    job_files.append(script_path)
                    total_jobs += 1

    return job_files, total_jobs


def generate_array_jobs(models, tasks, seeds, output_dir, concurrent=10):
    """Generate SLURM array job scripts (one per task)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    job_files = []

    for task in tasks:
        task_config = TASK_CONFIGS[task]
        seq_lens = task_config['seq_lens']

        total = len(models) * len(seq_lens) * len(seeds)
        max_index = total - 1

        # Estimate max time based on slowest model
        max_time = max(MODEL_CONFIGS[m]['time'] for m in models)

        script_content = ARRAY_TEMPLATE.format(
            task=task,
            models=' '.join(models),
            seq_lens=' '.join(map(str, seq_lens)),
            seeds=' '.join(map(str, seeds)),
            max_time=max_time,
            max_index=max_index,
            concurrent=concurrent,
            total=total,
            output_dir=str(output_dir)
        )

        script_path = output_dir / f"synthetic_{task}_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        job_files.append(script_path)

    return job_files


def generate_submit_all_script(output_dir, job_type='individual'):
    """Generate a script to submit all jobs."""
    output_dir = Path(output_dir)

    if job_type == 'individual':
        script_content = '''#!/bin/bash
# Submit all individual synthetic benchmark jobs

cd {output_dir}

for script in syn_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
    sleep 0.5  # Small delay to avoid overwhelming scheduler
done

echo "All jobs submitted!"
'''.format(output_dir=str(output_dir))
    else:
        script_content = '''#!/bin/bash
# Submit all synthetic benchmark array jobs

cd {output_dir}

for task in forrelation adding_problem selective_copy; do
    script="synthetic_${{task}}_array.sh"
    if [ -f "$script" ]; then
        echo "Submitting $script"
        sbatch "$script"
        sleep 1
    fi
done

echo "All array jobs submitted!"
'''.format(output_dir=str(output_dir))

    script_path = output_dir / "submit_all.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM job scripts for synthetic benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Default output directory from config
    default_output_dir = str(get_jobs_dir('synthetic'))
    
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help=f"Output directory for SLURM scripts (default: {default_output_dir})")
    parser.add_argument("--models", nargs='+',
                        default=['1a', '1b', '1c', '2a', '2d', '2e', '3a', '3b', '3c', '4a', '4d', '4e'],
                        help="Models to include")
    parser.add_argument("--tasks", nargs='+',
                        default=['forrelation', 'adding_problem', 'selective_copy'],
                        help="Tasks to include")
    parser.add_argument("--seeds", nargs='+', type=int,
                        default=DEFAULT_SEEDS,
                        help="Random seeds")
    parser.add_argument("--job-type", type=str, choices=['individual', 'array'], default='individual',
                        help="Type of job scripts to generate")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Max concurrent array jobs (for array type)")

    args = parser.parse_args()

    print("=" * 60)
    print("Generating SLURM Job Scripts for Synthetic Benchmarks")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Tasks: {args.tasks}")
    print(f"Seeds: {args.seeds}")
    print(f"Job type: {args.job_type}")
    print(f"Output dir: {args.output_dir}")
    print("-" * 60)

    if args.job_type == 'individual':
        job_files, total = generate_individual_jobs(
            args.models, args.tasks, args.seeds, args.output_dir
        )
        print(f"\nGenerated {total} individual job scripts")
    else:
        job_files = generate_array_jobs(
            args.models, args.tasks, args.seeds, args.output_dir, args.concurrent
        )
        print(f"\nGenerated {len(job_files)} array job scripts")

    submit_script = generate_submit_all_script(args.output_dir, args.job_type)
    print(f"Generated submit script: {submit_script}")

    # Calculate totals
    total_experiments = 0
    for task in args.tasks:
        task_config = TASK_CONFIGS[task]
        total_experiments += len(args.models) * len(task_config['seq_lens']) * len(args.seeds)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total experiments: {total_experiments}")
    print(f"  - Models: {len(args.models)}")
    print(f"  - Tasks: {len(args.tasks)}")
    print(f"  - Seeds: {len(args.seeds)}")

    print("\nTo submit all jobs:")
    print(f"  cd {args.output_dir}")
    print(f"  ./submit_all.sh")

    # Also print breakdown by task
    print("\nBreakdown by task:")
    for task in args.tasks:
        task_config = TASK_CONFIGS[task]
        n_exp = len(args.models) * len(task_config['seq_lens']) * len(args.seeds)
        print(f"  {task}: {n_exp} experiments (seq_lens: {task_config['seq_lens']})")


if __name__ == "__main__":
    main()
