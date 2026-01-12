#!/bin/bash
#SBATCH --job-name=abl_1b_80Hz_s2024
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./results/ablation_eeg/logs/abl_1b_80Hz_s2024.log
#SBATCH --error=./results/ablation_eeg/logs/abl_1b_80Hz_s2024.log

# ============================================
# Ablation Study Job [40/108]
# ============================================
# Model: 1b (QuantumMambaSSM)
# Sampling Freq: 80 Hz
# Seed: 2024
# ============================================

echo "============================================"
echo "Ablation Study - EEG Classification"
echo "============================================"
echo "Job: abl_1b_80Hz_s2024"
echo "Model: 1b (QuantumMambaSSM)"
echo "Sampling Freq: 80 Hz"
echo "Seed: 2024"
echo "Started: $(date)"
echo "============================================"
# Navigate to project root
cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba

# Run training (with  to automatically continue from checkpoint if available)
/scratch/connectome/mandy/envs/qhydra/bin/python scripts/run_ablation_eeg.py \
    --model-id 1b \
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
    --seed 2024 \
    --output-dir ./results/ablation_eeg \
    --device cuda \
    

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
