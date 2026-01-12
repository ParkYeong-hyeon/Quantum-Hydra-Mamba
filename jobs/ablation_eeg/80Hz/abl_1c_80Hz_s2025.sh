#!/bin/bash
#SBATCH --job-name=abl_1c_80Hz_s2025
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./results/ablation_eeg/logs/abl_1c_80Hz_s2025.log
#SBATCH --error=./results/ablation_eeg/logs/abl_1c_80Hz_s2025.log

# ============================================
# Ablation Study Job [44/108]
# ============================================
# Model: 1c (QuantumHydraSSM)
# Sampling Freq: 80 Hz
# Seed: 2025
# ============================================

echo "============================================"
echo "Ablation Study - EEG Classification"
echo "============================================"
echo "Job: abl_1c_80Hz_s2025"
echo "Model: 1c (QuantumHydraSSM)"
echo "Sampling Freq: 80 Hz"
echo "Seed: 2025"
echo "Started: $(date)"
echo "============================================"
# Navigate to project root
cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba

# Run training (with  to automatically continue from checkpoint if available)
/scratch/connectome/mandy/envs/qhydra/bin/python scripts/run_ablation_eeg.py \
    --model-id 1c \
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
    --sampling-freq 80 \
    --seed 2025 \
    --output-dir ./results/ablation_eeg \
    --device cuda \
    

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
