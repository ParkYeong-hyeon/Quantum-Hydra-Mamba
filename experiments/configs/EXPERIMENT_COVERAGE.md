# Experiment Configuration Coverage

이 문서는 `experiments/configs/`에 있는 설정 파일들이 실제로 시행된 모든 실험을 포함하는지 확인합니다.

## 확인 결과 요약

✅ **모든 실험이 포함되어 있습니다!**

## 상세 확인

### Ablation Study (`ablation/`)

| 실험 타입 | Job 위치 | Config 파일 | 상태 |
|----------|---------|------------|------|
| Synthetic Benchmarks | `jobs/synthetic/` | `ablation/synthetic.yaml` | ✅ |
| EEG Ablation Study | `jobs/ablation_eeg/` | `ablation/eeg.yaml` | ✅ |

**Job Generation Scripts**:
- `scripts/job_generation/generate_synthetic_jobs.py` → `ablation/synthetic.yaml` ✅
- `scripts/job_generation/generate_ablation_eeg_jobs.py` → `ablation/eeg.yaml` ✅

### Legacy Experiments (`legacy/`)

| 실험 타입 | Job 위치 | Config 파일 | 상태 |
|----------|---------|------------|------|
| EEG | `job_scripts/eeg/` | `legacy/eeg.yaml` | ✅ |
| DNA | `job_scripts/dna/` | `legacy/dna.yaml` | ✅ |
| MNIST | `job_scripts/image/` | `legacy/mnist.yaml` | ✅ |
| Gated Models | `job_scripts/gated/` | `legacy/gated.yaml` | ✅ |
| Genomic | `job_scripts/genomic/` | `legacy/genomic.yaml` | ✅ |
| GLUE | `job_scripts/glue/` | `legacy/glue.yaml` | ✅ |
| Forrelation | `job_scripts/forrelation/` | `legacy/forrelation.yaml` | ✅ (추가됨) |
| Full Comparison | `job_scripts/full_comparison.sh` | `legacy/full_comparison.yaml` | ✅ |

**Job Generation Scripts**:
- `scripts/job_generation/generate_eeg_job_scripts.py` → `legacy/eeg.yaml` ✅
- `scripts/job_generation/generate_dna_job_scripts.py` → `legacy/dna.yaml` ✅
- `scripts/job_generation/generate_mnist_job_scripts.py` → `legacy/mnist.yaml` ✅
- `scripts/job_generation/generate_forrelation_job_scripts.py` → `legacy/forrelation.yaml` ✅ (추가됨)

## 실험 통계

### Ablation Study
- **Synthetic**: 12 models × 3 tasks × 4 seq_lens × 3 seeds = **432 experiments**
- **EEG**: 14 models × 3 sampling_freqs × 3 seeds = **126 experiments**
- **Total Ablation**: **558 experiments**

### Legacy Experiments
- **EEG**: 6 models × 5 seeds = **30 experiments**
- **DNA**: 6 models × 5 seeds = **30 experiments**
- **MNIST**: 6 models × 5 seeds = **30 experiments**
- **Forrelation**: 6 models × 8 datasets × 3 seeds = **144 experiments**
- **Gated**: 2 models × 5 seeds × 3 datasets = **30 experiments** (EEG, DNA, Genomic)
- **Genomic**: Variable (SSM comparison, gated models)
- **GLUE**: 9 tasks × multiple models
- **Full Comparison**: 6 models × 3 datasets × 3 seeds = **54 experiments**
- **Total Legacy**: **~300+ experiments**

## 파일 구조 매핑

```
실제 Job 위치                    → Config 파일
─────────────────────────────────────────────────────
jobs/synthetic/                  → ablation/synthetic.yaml
jobs/ablation_eeg/               → ablation/eeg.yaml
job_scripts/eeg/                 → legacy/eeg.yaml
job_scripts/dna/                 → legacy/dna.yaml
job_scripts/image/ (MNIST)       → legacy/mnist.yaml
job_scripts/gated/                → legacy/gated.yaml
job_scripts/genomic/             → legacy/genomic.yaml
job_scripts/glue/                → legacy/glue.yaml
job_scripts/forrelation/         → legacy/forrelation.yaml
job_scripts/full_comparison.sh   → legacy/full_comparison.yaml
```

## 누락된 실험

**없음** ✅

모든 실험이 설정 파일에 포함되어 있습니다.

## 추가 확인 사항

### 특수 케이스

1. **EEG 변형 실험**: `job_scripts/eeg/`에 `10q`, `8q`, `160hz` 등의 변형이 있지만, 이들은 기본 `legacy/eeg.yaml`의 하이퍼파라미터 변형으로 볼 수 있습니다.

2. **Gated 모델의 다양한 데이터셋**: `legacy/gated.yaml`에 EEG, DNA, Genomic이 모두 포함되어 있습니다.

3. **Forrelation Phase 3/4**: `legacy/forrelation.yaml`에 Phase 3 (Silver Standard)와 Phase 4 (Gold Standard)가 모두 포함되어 있습니다.

## 권장 사항

1. ✅ 모든 실험이 설정 파일에 포함되어 있음
2. ✅ Legacy와 Ablation이 명확히 분리되어 있음
3. ✅ 각 설정 파일이 해당 job_scripts/jobs와 일치함

## 업데이트 이력

- **2025-01-XX**: 초기 확인 완료
- **2025-01-XX**: `legacy/forrelation.yaml` 추가 (누락 발견 및 수정)
