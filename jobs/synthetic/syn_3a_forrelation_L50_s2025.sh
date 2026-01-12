#!/bin/bash
#SBATCH --job-name=syn_3a_forrelation_L50_s2025
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/jobs/synthetic/logs/syn_3a_forrelation_L50_s2025_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/jobs/synthetic/logs/syn_3a_forrelation_L50_s2025_%j.err

# ============================================
# Synthetic Benchmark Experiment
# ============================================
# Model: 3a (ClassicalTransformer)
# Task: forrelation
# Sequence Length: 50
# Seed: 2025
# ============================================

echo "Starting job: syn_3a_forrelation_L50_s2025"
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
    --model-id 3a \
    --task forrelation \
    --seq-len 50 \
    --seed 2025 \
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

echo "Job completed: syn_3a_forrelation_L50_s2025"
echo "End time: $(date)"
