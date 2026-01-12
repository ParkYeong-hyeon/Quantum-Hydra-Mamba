#!/bin/bash
#SBATCH --job-name=abl_2c_160Hz_s2026
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./results/ablation_eeg/logs/abl_2c_160Hz_s2026.log
#SBATCH --error=./results/ablation_eeg/logs/abl_2c_160Hz_s2026.log

# ============================================
# Ablation Study Job [90/108]
# ============================================
# Model: 2c (ClassicalHydraQuantumSSM)
# Sampling Freq: 160 Hz
# Seed: 2026
# ============================================

echo "============================================"
echo "Ablation Study - EEG Classification"
echo "============================================"
echo "Job: abl_2c_160Hz_s2026"
echo "Model: 2c (ClassicalHydraQuantumSSM)"
echo "Sampling Freq: 160 Hz"
echo "Seed: 2026"
echo "Started: $(date)"
echo "============================================"
# Navigate to project root
cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba

# Run training (with --resume to automatically continue from checkpoint if available)
/scratch/connectome/mandy/envs/qhydra/bin/python scripts/run_ablation_eeg.py \
    --model-id 2c \
    --n-qubits 6 \
    --n-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --early-stopping 10 \
    --sample-size 109 \
    --sampling-freq 160 \
    --seed 2026 \
    --output-dir ./results/ablation_eeg \
    --device cuda \
    --resume

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
