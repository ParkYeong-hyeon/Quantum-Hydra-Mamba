# Quantum Hydra & Quantum Mamba Experiment Status Report
**Generated:** 2025-01-15
**Total Experiments:** 90 (3 datasets × 6 models × 5 seeds)

---

## Executive Summary

**✅ Successful:** 57/90 (63.3%)
**❌ Failed with crash:** 18/90 (20.0%)
**⚠️  Completed with NaN:** 5/90 (5.6%)
**❌ Never ran:** 10/90 (11.1%)

---

## Detailed Breakdown by Dataset

### 1. DNA Dataset (22/30 successful)

| Model | Status | Details |
|-------|--------|---------|
| `quantum_hydra` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_hydra_hybrid` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_mamba` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_mamba_hybrid` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `classical_hydra` | ❌ **0/5 FAILED** | All seeds crashed with IndexError |
| `classical_mamba` | ⚠️  **2/5 SUCCESS** | Seeds 2024-2026 crashed; 2027-2028 succeeded |

**Failed Experiments:**
- `dna_classical_hydra_seed2024` through `seed2028` (5 failures)
- `dna_classical_mamba_seed2024` through `seed2026` (3 failures)

**Error:** `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)`
**Location:** `x.transpose(1, 2)` in TrueClassicalHydra.py:376 and TrueClassicalMamba.py:295

---

### 2. MNIST Dataset (20/30 successful)

| Model | Status | Details |
|-------|--------|---------|
| `quantum_hydra` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_hydra_hybrid` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_mamba` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_mamba_hybrid` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `classical_hydra` | ❌ **0/5 FAILED** | All seeds crashed with IndexError |
| `classical_mamba` | ❌ **0/5 FAILED** | All seeds crashed with IndexError |

**Failed Experiments:**
- `mnist_classical_hydra_seed2024` through `seed2028` (5 failures)
- `mnist_classical_mamba_seed2024` through `seed2028` (5 failures)

**Error:** `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)`
**Location:** Same as DNA dataset

---

### 3. EEG Dataset (15/30 successful, 5 NaN issues, 10 never ran)

| Model | Status | Details |
|-------|--------|---------|
| `quantum_hydra` | ❌ **0/5 NEVER RAN** | No logs generated |
| `quantum_hydra_hybrid` | ❌ **0/5 NEVER RAN** | No logs generated |
| `quantum_mamba` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `quantum_mamba_hybrid` | ✅ **5/5 SUCCESS** | All seeds completed successfully |
| `classical_hydra` | ⚠️  **5/5 NaN LOSS** | Completed but with NaN loss/poor accuracy |
| `classical_mamba` | ✅ **5/5 SUCCESS** | All seeds completed successfully |

**Never Ran (10 experiments):**
- `eeg_quantum_hydra_seed2024` through `seed2028` (5 missing)
- `eeg_quantum_hydra_hybrid_seed2024` through `seed2028` (5 missing)

**NaN Loss Issues (5 experiments):**
- `eeg_classical_hydra_seed2024` through `seed2028`
- Training loss became NaN, accuracy ~50% (random chance)
- Jobs completed but results are invalid

---

## Critical Issues Identified

### Issue #1: IndexError in Classical Models (18 failures)

**Affected Models:** `TrueClassicalHydra.py`, `TrueClassicalMamba.py`
**Datasets:** DNA (8 failures), MNIST (10 failures)
**Seeds:** DNA: 2024-2028 for Hydra, 2024-2026 for Mamba; MNIST: 2024-2028 for both

**Error Message:**
```
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
File "/pscratch/sd/j/junghoon/TrueClassicalHydra.py", line 376, in forward
    x = x.transpose(1, 2)
File "/pscratch/sd/j/junghoon/TrueClassicalMamba.py", line 295, in forward
    x = x.transpose(1, 2)
```

**Root Cause:** Input tensor has only 2 dimensions, but code expects 3 dimensions (batch, seq, features)

**Fix Required:**
1. Check input dimensions before transpose
2. Handle 2D inputs appropriately
3. Add dimension validation

---

### Issue #2: EEG quantum_hydra Jobs Never Started (10 missing)

**Affected Models:** `quantum_hydra`, `quantum_hydra_hybrid`
**Dataset:** EEG only
**Seeds:** All (2024-2028)

**Problem:** No log files generated (neither SLURM nor Python logs)

**Possible Causes:**
- SLURM jobs failed to start (resource constraints?)
- Jobs were never submitted
- Jobs failed immediately before logging started
- Incorrect job submission parameters

**Investigation Needed:**
1. Check SLURM queue history: `sacct -u junghoon --starttime=2024-10-01`
2. Verify job submission scripts exist
3. Check for resource/permission issues

---

### Issue #3: EEG classical_hydra NaN Loss (5 completed but invalid)

**Affected Model:** `TrueClassicalHydra.py`
**Dataset:** EEG only
**Seeds:** All (2024-2028)

**Problem:** Training completed but loss became NaN, accuracy ~50% (random)

**Symptoms:**
```
Train Loss: nan, Train Acc: 0.5017
Val Loss: nan, Val Acc: 0.4983
Test Acc: 0.5058, Test AUC: 0.0000
```

**Possible Causes:**
- Exploding gradients
- Invalid learning rate
- Numerical instability in EEG data processing
- Incorrect normalization

**Fix Required:**
1. Add gradient clipping
2. Adjust learning rate
3. Add NaN checks during training
4. Investigate EEG data preprocessing

---

## Action Items

### Priority 1: Fix Classical Model Dimension Error (18 experiments)

**Files to fix:**
- `/pscratch/sd/j/junghoon/quantum_hydra_mamba_repo/models/TrueClassicalHydra.py:376`
- `/pscratch/sd/j/junghoon/quantum_hydra_mamba_repo/models/TrueClassicalMamba.py:295`

**Steps:**
1. Examine input tensor shapes for DNA and MNIST datasets
2. Update `forward()` method to handle 2D inputs
3. Add dimension checks and reshape if needed
4. Re-run failed experiments

**Experiments to re-run (18 total):**
```bash
# DNA (8 experiments)
dna_classical_hydra_seed{2024,2025,2026,2027,2028}
dna_classical_mamba_seed{2024,2025,2026}

# MNIST (10 experiments)
mnist_classical_hydra_seed{2024,2025,2026,2027,2028}
mnist_classical_mamba_seed{2024,2025,2026,2027,2028}
```

---

### Priority 2: Investigate Missing EEG quantum_hydra Jobs (10 experiments)

**Files to check:**
- Job submission scripts for EEG quantum_hydra models
- SLURM job history

**Steps:**
1. Verify job scripts exist and are correct
2. Check SLURM history for job submission failures
3. Submit or re-submit jobs if needed

**Missing experiments (10 total):**
```bash
# EEG quantum_hydra (5 experiments)
eeg_quantum_hydra_seed{2024,2025,2026,2027,2028}

# EEG quantum_hydra_hybrid (5 experiments)
eeg_quantum_hydra_hybrid_seed{2024,2025,2026,2027,2028}
```

---

### Priority 3: Fix EEG classical_hydra NaN Issue (5 experiments)

**File to fix:**
- `/pscratch/sd/j/junghoon/quantum_hydra_mamba_repo/models/TrueClassicalHydra.py`

**Steps:**
1. Add gradient clipping (torch.nn.utils.clip_grad_norm_)
2. Reduce learning rate for EEG dataset
3. Add NaN detection and early stopping
4. Check data normalization for EEG

**Experiments to re-run (5 total):**
```bash
eeg_classical_hydra_seed{2024,2025,2026,2027,2028}
```

---

## Summary Statistics

### By Model Type

| Model Type | Success Rate | Notes |
|------------|--------------|-------|
| `quantum_hydra` | 10/15 (67%) | EEG never ran (0/5) |
| `quantum_hydra_hybrid` | 10/15 (67%) | EEG never ran (0/5) |
| `quantum_mamba` | 15/15 (100%) | ✅ All successful |
| `quantum_mamba_hybrid` | 15/15 (100%) | ✅ All successful |
| `classical_hydra` | 0/15 (0%) | DNA/MNIST crashed, EEG NaN |
| `classical_mamba` | 12/15 (80%) | DNA partial, MNIST all failed |

### By Dataset

| Dataset | Success Rate | Issues |
|---------|--------------|--------|
| DNA | 22/30 (73%) | Classical models crashed (8 failures) |
| MNIST | 20/30 (67%) | Classical models all crashed (10 failures) |
| EEG | 15/30 (50%) | 10 never ran, 5 NaN issues |

---

## Next Steps

1. **Immediate:** Fix the dimension error in classical models (Priority 1)
2. **Soon:** Investigate why EEG quantum_hydra jobs never started (Priority 2)
3. **After fixes:** Re-run all failed experiments (33 total)
4. **Optional:** Fix EEG classical_hydra NaN issue (Priority 3)

**Expected final status after fixes:** 85/90 successful (94.4%)
- 57 currently successful
- 18 will be fixed (IndexError)
- 10 will be run (never started)
- 5 may remain with NaN issues (requires deeper investigation)

---

## Files for Reference

- Analysis script: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/analyze_all_experiments.py`
- Model files: `/pscratch/sd/j/junghoon/quantum_hydra_mamba_repo/models/`
- Experiment logs: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/{dna,mnist,eeg}_results/logs/`
- Results: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/{dna,mnist,eeg}_results/`
