#!/bin/bash
#SBATCH --job-name=syn_4a_forrelation_L100_s2026
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/jobs/synthetic/logs/syn_4a_forrelation_L100_s2026_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/jobs/synthetic/logs/syn_4a_forrelation_L100_s2026_%j.err

# ============================================
# Synthetic Benchmark Experiment
# ============================================
# Model: 4a (QuantumTransformerE2E)
# Task: forrelation
# Sequence Length: 100
# Seed: 2026
# ============================================

echo "Starting job: syn_4a_forrelation_L100_s2026"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Navigate to project directory
cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export PENNYLANE_DEVICE=default.qubit

# Run experiment
/scratch/connectome/mandy/envs/qhydra/bin/python scripts/run_synthetic_benchmark.py \
    --model-id 4a \
    --task forrelation \
    --seq-len 100 \
    --seed 2026 \
    --n-qubits 6 \
    --n-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --early-stopping 20 \
    --num-markers 8 \
    --output-dir ./results/synthetic_benchmarks \
    --data-dir ./data/synthetic_benchmarks \
    --device cuda

echo "Job completed: syn_4a_forrelation_L100_s2026"
echo "End time: $(date)"
