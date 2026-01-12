#!/bin/bash
# ============================================
# Submit all jobs for Model 2d (QuantumMambaHydraSSM)
# True Quantum Superposition + Delta Recurrence
# ============================================
# 3 frequencies × 3 seeds = 9 jobs
# ============================================

echo "Submitting Model 2d (QuantumMambaHydraSSM) jobs..."
echo "  - True quantum superposition (3 branches)"
echo "  - Delta-modulated selective forgetting"
echo "  - Unidirectional processing"
echo ""

cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba/jobs/ablation_eeg

# 40Hz jobs
echo "Submitting 40Hz jobs..."
sbatch 40Hz/abl_2d_40Hz_s2024.sh
sbatch 40Hz/abl_2d_40Hz_s2025.sh
sbatch 40Hz/abl_2d_40Hz_s2026.sh

# 80Hz jobs
echo "Submitting 80Hz jobs..."
sbatch 80Hz/abl_2d_80Hz_s2024.sh
sbatch 80Hz/abl_2d_80Hz_s2025.sh
sbatch 80Hz/abl_2d_80Hz_s2026.sh

# 160Hz jobs
echo "Submitting 160Hz jobs..."
sbatch 160Hz/abl_2d_160Hz_s2024.sh
sbatch 160Hz/abl_2d_160Hz_s2025.sh
sbatch 160Hz/abl_2d_160Hz_s2026.sh

echo ""
echo "Submitted 9 jobs for Model 2d"
echo "Check status with: squeue -u $USER"
