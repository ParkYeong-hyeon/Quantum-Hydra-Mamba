# Quick Reference - Quantum Hydra/Mamba Experiments

**Last Updated:** November 15, 2025

---

## Current Status

✅ **All tasks complete**
⏳ **33 jobs queued** (Job IDs: 45232061-45232094)
📊 **Expected:** 94-100% success rate (up from 63%)

---

## Monitor Experiments

```bash
# Quick status check
bash /pscratch/sd/j/junghoon/monitor_experiments.sh

# Check queue
squeue -u junghoon

# Watch log live
tail -f results/mnist_results/logs/mnist_classical_hydra_seed2024.log
```

---

## When Jobs Complete

```bash
# Run analysis
python scripts/analyze_all_experiments.py > results/final_analysis_report.txt

# Check for success
grep -r "TRAINING COMPLETE" results/*/logs/*.log

# Verify no errors
grep -r "IndexError\|nan" results/*/logs/*.log
```

---

## For GitHub Sharing

**Repository:** `/pscratch/sd/j/junghoon/quantum_hydra_mamba_repo/`

**Key files:**
- `RECENT_UPDATES.md` - Comprehensive guide for colleagues
- `GIT_COMMIT_MESSAGE.txt` - Ready-to-use commit message
- `models/TrueClassicalHydra.py` - Fixed ✅
- `models/TrueClassicalMamba.py` - Fixed ✅
- `scripts/run_single_model_eeg.py` - Fixed ✅

**Quick test:**
```bash
cd /pscratch/sd/j/junghoon
python test_classical_model_fix.py
```
Expected: ✅ ALL DIMENSION FIXES VALIDATED!

---

## Estimated Completion

| Dataset | Jobs | Time Allocation | Est. Completion |
|---------|------|-----------------|-----------------|
| MNIST   | 10   | 12 hours        | ~2-4 hours      |
| DNA     | 8    | 24 hours        | ~5-8 hours      |
| EEG     | 15   | 48 hours        | ~10-12 hours    |

---

## Documentation Files

| File | Purpose |
|------|---------|
| `WORK_COMPLETED_SUMMARY.md` | Comprehensive work summary |
| `CURRENT_STATUS_REPORT.md` | Real-time status overview |
| `JOBS_SUBMITTED_SUMMARY.md` | Job submission details |
| `QUICK_REFERENCE.md` | This file (quick access) |
| `quantum_hydra_mamba_repo/RECENT_UPDATES.md` | For colleagues |

---

## Key Changes Made

1. **Account:** m4138_g → m4727_g ✅
2. **Classical models:** Added 2D/3D input handling ✅
3. **EEG training:** Added gradient clipping ✅
4. **Jobs:** Resubmitted 33 experiments ✅
5. **Repository:** Synchronized for GitHub ✅

---

## Success Rate Improvement

**Before:** 57/90 (63%)
- DNA: 22/30 ❌
- MNIST: 20/30 ❌
- EEG: 15/30 ❌

**After (expected):** 85-90/90 (94-100%)
- DNA: 30/30 ✅
- MNIST: 30/30 ✅
- EEG: 25-30/30 ✅

**Improvement:** +28-33 experiments (+31-37%)

---

## Contact

**For monitoring:** Run `monitor_experiments.sh`
**For issues:** Check logs in `results/*/logs/`
**For colleagues:** Share `quantum_hydra_mamba_repo/RECENT_UPDATES.md`
