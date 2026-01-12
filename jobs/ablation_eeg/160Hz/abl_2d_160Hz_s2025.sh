#!/bin/bash
#SBATCH --job-name=abl_2d_160Hz_s2025
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/ablation_eeg/logs/abl_2d_160Hz_s2025.log
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/ablation_eeg/logs/abl_2d_160Hz_s2025.log

# ============================================
# Ablation Study: NEW Quantum Hydra SSM
# ============================================
# Model: 2d (QuantumMambaHydraSSM)
# Sampling Freq: 160 Hz
# Seed: 2025
# ============================================

echo "============================================"
echo "Ablation Study - EEG Classification"
echo "============================================"
echo "Job: abl_2d_160Hz_s2025"
echo "Model: 2d (QuantumMambaHydraSSM)"
echo "Sampling Freq: 160 Hz"
echo "Seed: 2025"
echo "Started: $(date)"
echo "============================================"
# Navigate to project root
cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba

# Run training (with --resume to automatically continue from checkpoint if available)
/scratch/connectome/mandy/envs/qhydra/bin/python scripts/run_ablation_eeg.py \
    --model-id 2d \
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
    --seed 2025 \
    --output-dir ./results/ablation_eeg \
    --device cuda \
    --resume

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
