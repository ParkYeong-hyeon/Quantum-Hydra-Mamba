# Documentation Index

This directory contains comprehensive documentation for the Quantum Hydra and Quantum Mamba models, organized by category for easy navigation.

---

## 🚀 Quick Start

**New to the project?** Start here:
1. **[Experimental Guide](../02_experimental_plans/EXPERIMENT_GUIDE.md)** ⭐ - Complete guide for running experiments
2. **[Architecture Guides](#-guide-documents)** - Understand model structures
3. **[Results](../03_results/)** - See experimental results

---

## 📚 Guide Documents

All essential guides in one place for quick reference:

### 🏗️ Architecture Guides
**Understanding model structures and implementation details**

#### Legacy Models (Initial Experiments)
- **[QUANTUM_HYDRA_GUIDE.md](../01_architecture/legacy/QUANTUM_HYDRA_GUIDE.md)** - Quantum Hydra models (Option A vs Option B)
- **[QUANTUM_MAMBA_GUIDE.md](../01_architecture/legacy/QUANTUM_MAMBA_GUIDE.md)** - Quantum Mamba models (Option A vs Option B)
- **[QUANTUM_GATED_RECURRENCE_GUIDE.md](../01_architecture/legacy/QUANTUM_GATED_RECURRENCE_GUIDE.md)** - Gated recurrence models (Legacy extension)
- **[FOUR_MODEL_ARCHITECTURE_COMPARISON.md](../01_architecture/legacy/FOUR_MODEL_ARCHITECTURE_COMPARISON.md)** - Comparison of 4 model architectures

#### Ablation Study Models
- **[QUANTUM_SSM_README.md](../01_architecture/ablation/QUANTUM_SSM_README.md)** - QuantumMambaSSM and QuantumHydraSSM detailed guide

#### Shared/Common
- **[QUANTUM_RECURRENCE_DISCUSSION.md](../01_architecture/shared/QUANTUM_RECURRENCE_DISCUSSION.md)** - Quantum recurrence mechanisms
- **[QUANTUM_SEQUENCE_PROCESSING.md](../01_architecture/shared/QUANTUM_SEQUENCE_PROCESSING.md)** - Quantum sequence processing theory

### 🧪 Experimental Guides
**How to design and run experiments**

#### Ablation Study (Current)
- **[EXPERIMENT_GUIDE.md](../02_experimental_plans/ablation/EXPERIMENT_GUIDE.md)** ⭐ **START HERE** - Complete experimental framework
- **[ABLATION_STUDY_PLAN_V3.md](../02_experimental_plans/ablation/ABLATION_STUDY_PLAN_V3.md)** - 2×2×3 Factorial Design (Latest)
- **[SYNTHETIC_BENCHMARK_PROTOCOL.md](../02_experimental_plans/ablation/SYNTHETIC_BENCHMARK_PROTOCOL.md)** - Synthetic benchmark protocol
- **[Quantum_Advantage_Test_Plan.md](../02_experimental_plans/ablation/Quantum_Advantage_Test_Plan.md)** - Quantum advantage testing protocol

#### Legacy Experiments
- **[QuantumHydra_ExperimentPlan_README.md](../02_experimental_plans/legacy/QuantumHydra_ExperimentPlan_README.md)** - Legacy Quantum Hydra experiment plan

#### General
- **[EXPERIMENTAL_PLAN_README.md](../02_experimental_plans/EXPERIMENTAL_PLAN_README.md)** - General experimental plans

### 📊 Dataset Guides
**Setting up and using datasets**

- **[EEG_DATASETS_SETUP_GUIDE.md](../04_datasets/EEG_DATASETS_SETUP_GUIDE.md)** - EEG dataset setup and usage
- **[FORRELATION_README.md](../04_datasets/FORRELATION_README.md)** - Forrelation experiment guide
- **[Forrelation_Dataset_Usage_Guide.md](../04_datasets/Forrelation_Dataset_Usage_Guide.md)** - Forrelation dataset practical guide
- **[Forrelation_Experiment_Rationale.md](../04_datasets/Forrelation_Experiment_Rationale.md)** - Forrelation theoretical background

### 📈 Analysis Guides
**Understanding results and metrics**

#### Shared/Common Analysis
- **[RESEARCH_QUESTIONS_ANSWERS.md](../06_analysis/shared/RESEARCH_QUESTIONS_ANSWERS.md)** ⭐ - Answers to key research questions
- **[TIMING_AND_METRICS.md](../06_analysis/shared/TIMING_AND_METRICS.md)** - Timing information and performance metrics
- **[COMPUTATIONAL_COMPLEXITY_COMPARISON.md](../06_analysis/shared/COMPUTATIONAL_COMPLEXITY_COMPARISON.md)** - Computational complexity analysis
- **[MULTIGPU_QUANTUM_PARADOX.md](../06_analysis/shared/MULTIGPU_QUANTUM_PARADOX.md)** - Multi-GPU quantum paradox discussion

#### Legacy Analysis
- **[QUANTUM_GATED_COMPLEXITY_ANALYSIS.md](../06_analysis/legacy/QUANTUM_GATED_COMPLEXITY_ANALYSIS.md)** - Gated model complexity analysis

### 🎯 Benchmark Guides
**Running benchmark experiments**

- **[GLUE_README.md](../07_benchmarks/GLUE_README.md)** - GLUE benchmark guide
- **[EXPERIMENTS_README.md](../07_benchmarks/EXPERIMENTS_README.md)** - Experiments overview

---

## 📁 Documentation Structure

### 01. [Architecture Guides](../01_architecture/)
Model structures, design principles, and implementation details
- **[Legacy Models](../01_architecture/legacy/)** - Initial experimental models (Quantum Hydra, Quantum Mamba, Gated variants)
- **[Ablation Study Models](../01_architecture/ablation/)** - 2×2×3 factorial design models
- **[Shared/Common](../01_architecture/shared/)** - Common architecture discussions

### 02. [Experimental Plans](../02_experimental_plans/)
Experimental design, protocols, and execution guides
- **[Ablation Study](../02_experimental_plans/ablation/)** - 2×2×3 factorial design plans (V3 is latest)
- **[Legacy Experiments](../02_experimental_plans/legacy/)** - Initial experimental plans
- General experimental plans

### 03. [Results](../03_results/)
Experimental results, summaries, and performance analyses
- **[Ablation Study Results](../03_results/ablation/)** - 2×2×3 factorial design results
- **[Legacy Results](../03_results/legacy/)** - Initial experimental results (including Gated models)

### 04. [Datasets](../04_datasets/)
Dataset setup guides and usage instructions
- EEG dataset setup
- Forrelation dataset guides
- Dataset rationale

### 05. [Status Reports](../05_status_reports/)
Progress reports and completed work summaries
- Current status reports
- Work completion summaries
- Job submission summaries

### 06. [Analysis](../06_analysis/)
Deep analysis, research questions, and complexity studies
- **[Shared/Common Analysis](../06_analysis/shared/)** - Research questions, complexity comparisons, theoretical discussions
- **[Legacy Analysis](../06_analysis/legacy/)** - Legacy model-specific analysis

### 07. [Benchmarks](../07_benchmarks/)
Benchmark experiment guides
- GLUE benchmark
- Experiment overviews

### 08. [Reference](../08_reference/)
Legacy documents and quick references
- Old repository README
- Quick reference guides
- Configuration summaries

---

## 🎯 Quick Navigation

### I want to...

**Run my first experiment:**
→ Read [Experimental Guide](../02_experimental_plans/ablation/EXPERIMENT_GUIDE.md)
→ Run: `python scripts/training/run_ablation_eeg.py --model-id 1c --sampling-freq 80 --seed 2024`

**Understand model architectures:**
→ Read [Architecture Guides](../01_architecture/)
→ Legacy: [QUANTUM_HYDRA_GUIDE.md](../01_architecture/legacy/QUANTUM_HYDRA_GUIDE.md)
→ Ablation: [QUANTUM_SSM_README.md](../01_architecture/ablation/QUANTUM_SSM_README.md)

**See experimental results:**
→ Browse [Results](../03_results/)
→ Ablation: [ABLATION_EEG_RESULTS_SUMMARY.md](../03_results/ablation/ABLATION_EEG_RESULTS_SUMMARY.md)
→ Legacy: [Legacy Results](../03_results/legacy/)

**Set up datasets:**
→ Read [Dataset Guides](../04_datasets/)
→ Start with [EEG_DATASETS_SETUP_GUIDE.md](../04_datasets/EEG_DATASETS_SETUP_GUIDE.md)

**Understand research findings:**
→ Read [Research Questions & Answers](../06_analysis/shared/RESEARCH_QUESTIONS_ANSWERS.md)
→ Check [Analysis](../06_analysis/) section

**Test quantum advantage:**
→ Read [FORRELATION_README.md](../04_datasets/FORRELATION_README.md)
→ Follow [Quantum Advantage Test Plan](../02_experimental_plans/ablation/Quantum_Advantage_Test_Plan.md)

---

## 📖 Recommended Reading Order

### For First-Time Users
1. **[EXPERIMENT_GUIDE.md](../02_experimental_plans/ablation/EXPERIMENT_GUIDE.md)** - Overall experimental framework
2. **[QUANTUM_SSM_README.md](../01_architecture/ablation/QUANTUM_SSM_README.md)** - Ablation study model architectures
3. **[QUANTUM_HYDRA_GUIDE.md](../01_architecture/legacy/QUANTUM_HYDRA_GUIDE.md)** - Legacy Quantum Hydra details
4. **[QUANTUM_MAMBA_GUIDE.md](../01_architecture/legacy/QUANTUM_MAMBA_GUIDE.md)** - Legacy Quantum Mamba details

### For Researchers
1. **[ABLATION_STUDY_PLAN_V3.md](../02_experimental_plans/ablation/ABLATION_STUDY_PLAN_V3.md)** - Experimental design
2. **[RESEARCH_QUESTIONS_ANSWERS.md](../06_analysis/shared/RESEARCH_QUESTIONS_ANSWERS.md)** - Key findings
3. **[ABLATION_EEG_RESULTS_SUMMARY.md](../03_results/ablation/ABLATION_EEG_RESULTS_SUMMARY.md)** - Results summary
4. Architecture guides for implementation details

### For Developers
1. **[QUANTUM_SSM_README.md](../01_architecture/ablation/QUANTUM_SSM_README.md)** - Ablation study implementation details
2. **[QUANTUM_HYDRA_GUIDE.md](../01_architecture/legacy/QUANTUM_HYDRA_GUIDE.md)** - Legacy Hydra implementation
3. **[QUANTUM_MAMBA_GUIDE.md](../01_architecture/legacy/QUANTUM_MAMBA_GUIDE.md)** - Legacy Mamba implementation
4. **[TIMING_AND_METRICS.md](../06_analysis/shared/TIMING_AND_METRICS.md)** - Performance analysis

---

## 🆕 Latest Updates

**Important:** This documentation covers the **2×2×3 factorial design ablation study** with 14 models.

**Latest Documents:**
- **[ABLATION_STUDY_PLAN_V3.md](../02_experimental_plans/ablation/ABLATION_STUDY_PLAN_V3.md)** - Latest ablation study plan
- **[RESEARCH_QUESTIONS_ANSWERS.md](../06_analysis/shared/RESEARCH_QUESTIONS_ANSWERS.md)** - Key research findings
- **[ABLATION_EEG_RESULTS_SUMMARY.md](../03_results/ablation/ABLATION_EEG_RESULTS_SUMMARY.md)** - Latest results

---

## 🔗 External Resources

**Classical Papers:**
- Hwang et al. (2024) - [Hydra: Bidirectional State Space Models](https://arxiv.org/pdf/2407.09941)
- Gu & Dao (2024) - [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/html/2312.00752v2)

**Quantum Advantage:**
- Aaronson & Ambainis (2015) - [Forrelation](https://arxiv.org/abs/1411.5729)

**Framework:**
- [PennyLane Documentation](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Last Updated:** January 2025
**Maintained by:** Junghoon Park
