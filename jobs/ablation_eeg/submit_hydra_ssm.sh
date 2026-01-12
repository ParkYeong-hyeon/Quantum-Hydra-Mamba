#!/bin/bash
# ============================================
# Submit all Quantum Hydra SSM jobs (Models 2d & 2e)
# ============================================
# NEW models: True Quantum Superposition + Delta Recurrence
# 2 models × 3 frequencies × 3 seeds = 18 jobs
# ============================================

echo "============================================"
echo "Submitting Quantum Hydra SSM Ablation Jobs"
echo "============================================"
echo ""
echo "Model 2d: QuantumMambaHydraSSM (unidirectional)"
echo "Model 2e: QuantumHydraHydraSSM (bidirectional)"
echo ""
echo "Key innovations:"
echo "  - TRUE quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩"
echo "  - Delta-modulated selective forgetting"
echo "  - EXACT same circuit structure as QuantumSSM.py"
echo ""
echo "============================================"

cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba/jobs/ablation_eeg

# Submit Model 2d jobs
echo ""
echo "[Model 2d - QuantumMambaHydraSSM]"
./submit_model_2d.sh

# Submit Model 2e jobs
echo ""
echo "[Model 2e - QuantumHydraHydraSSM]"
./submit_model_2e.sh

echo ""
echo "============================================"
echo "Total jobs submitted: 18"
echo "Check status with: squeue -u $USER"
echo "============================================"
