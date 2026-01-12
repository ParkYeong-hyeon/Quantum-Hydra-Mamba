#!/bin/bash
# Submit ALL 108 ablation study jobs
# WARNING: This will submit many jobs at once!

echo "Submitting 108 ablation study jobs..."
echo "Press Ctrl+C to cancel..."
sleep 3
# ===== 40 Hz Jobs =====
sbatch ./jobs/ablation_eeg/40Hz/abl_1a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2026.sh
sleep 1  # Pause between frequency batches

# ===== 80 Hz Jobs =====
sbatch ./jobs/ablation_eeg/80Hz/abl_1a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2026.sh
sleep 1  # Pause between frequency batches

# ===== 160 Hz Jobs =====
sbatch ./jobs/ablation_eeg/160Hz/abl_1a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1c_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3c_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2026.sh
sleep 1  # Pause between frequency batches

echo "============================================"
echo "All 108 jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Logs in: ./results/ablation_eeg/logs/"
echo "============================================"
