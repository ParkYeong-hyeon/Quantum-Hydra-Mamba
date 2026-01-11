# Documentation Organization Proposal

## 현재 상황 분석

현재 `docs/` 디렉토리는 **Legacy와 Ablation Study로 명시적으로 분리**되어 있습니다. 이 구조는 초기 실험 모델들과 체계적인 ablation study를 명확히 구분합니다.

### 분류 기준
- **Legacy**: 초기 실험 모델들 (quantum_hydra, quantum_mamba, quantum_hydra_hybrid, quantum_mamba_hybrid) + Gated 확장
- **Ablation**: 2×2×3 factorial design의 체계적 ablation study 모델들 (1a-4e)
- **Shared**: Legacy와 Ablation 모두에 적용되는 공통 문서

---

## 현재 구조 (리팩토링 완료)

```
docs/
├── 00_index/
│   └── README.md                    # 메인 인덱스
│
├── 01_architecture/
│   ├── legacy/                      # Legacy + Gated 통합
│   │   ├── QUANTUM_HYDRA_GUIDE.md
│   │   ├── QUANTUM_MAMBA_GUIDE.md
│   │   ├── QUANTUM_GATED_RECURRENCE_GUIDE.md
│   │   └── FOUR_MODEL_ARCHITECTURE_COMPARISON.md
│   ├── ablation/                    # Ablation Study
│   │   └── QUANTUM_SSM_README.md
│   └── shared/                      # 공통 문서
│       ├── QUANTUM_RECURRENCE_DISCUSSION.md
│       └── QUANTUM_SEQUENCE_PROCESSING.md
│
├── 02_experimental_plans/
│   ├── legacy/                      # Legacy 실험 계획
│   │   └── QuantumHydra_ExperimentPlan_README.md
│   ├── ablation/                    # Ablation Study 실험 계획
│   │   ├── ABLATION_STUDY_PLAN_V3.md                    # 최신 버전
│   │   ├── EXPERIMENT_GUIDE.md
│   │   ├── SYNTHETIC_BENCHMARK_PROTOCOL.md
│   │   ├── Quantum_Advantage_Test_Plan.md
│   │   └── legacy/                  # 이전 버전들
│   │       ├── ABLATION_STUDY_IMPLEMENTATION_PLAN.md
│   │       └── ABLATION_STUDY_IMPLEMENTATION_PLAN_V2.md
│   └── EXPERIMENTAL_PLAN_README.md  # 일반 실험 계획
│
├── 03_results/
│   ├── legacy/                      # Legacy 결과 (Gated 포함)
│   │   ├── QuantumHydraMamba_COMPREHENSIVE_RESULTS_2025-11-17.md
│   │   ├── EEG_EXPERIMENT_RESULTS_README.md
│   │   ├── DNA_CLASSIFICATION_RESULTS.md
│   │   ├── 8QUBIT_RESULTS_ANALYSIS.md
│   │   └── QUANTUM_GATED_RESULTS_SUMMARY.md
│   └── ablation/                    # Ablation Study 결과
│       ├── ABLATION_EEG_RESULTS_SUMMARY.md
│       └── GROUP2_PERFORMANCE_ANALYSIS.md
│
├── 04_datasets/
│   ├── EEG_DATASETS_SETUP_GUIDE.md
│   ├── Forrelation_Dataset_Usage_Guide.md
│   ├── FORRELATION_README.md
│   └── Forrelation_Experiment_Rationale.md
│
├── 05_status_reports/
│   ├── CURRENT_STATUS_REPORT.md
│   ├── EXPERIMENT_STATUS_REPORT.md
│   ├── WORK_COMPLETED_SUMMARY.md
│   ├── FIXES_COMPLETED_README.md
│   ├── JOBS_SUBMITTED_SUMMARY.md
│   └── TIER1_EXPERIMENTS_SUMMARY.md
│
├── 06_analysis/
│   ├── legacy/                      # Legacy 분석
│   │   └── QUANTUM_GATED_COMPLEXITY_ANALYSIS.md
│   └── shared/                      # 공통 분석
│       ├── RESEARCH_QUESTIONS_ANSWERS.md
│       ├── COMPUTATIONAL_COMPLEXITY_COMPARISON.md
│       ├── MULTIGPU_QUANTUM_PARADOX.md
│       └── TIMING_AND_METRICS.md
│
├── 07_benchmarks/
│   ├── GLUE_README.md
│   └── EXPERIMENTS_README.md
│
└── 08_reference/
    ├── OLD_REPO_README.md
    ├── QUICK_REFERENCE.md
    ├── UPDATED_CONFIGURATION_SUMMARY.md
    ├── OPTION_B_COMPLETE_SUMMARY.md
    └── README.md (현재 메인 README는 00_index로 이동)
```

---

## 카테고리별 설명

### 1. `00_index/` - 인덱스 및 시작점
**목적**: 프로젝트 문서의 시작점, 전체 구조 안내

**파일**:
- `README.md`: 메인 인덱스 (현재 docs/README.md)

**특징**:
- 모든 카테고리로의 링크 제공
- 신규 사용자를 위한 시작 가이드
- 문서 간 네비게이션

---

### 2. `01_architecture/` - 아키텍처 가이드
**목적**: 모델 구조, 설계 원칙, 구현 세부사항 설명

**구조**:
- `legacy/`: Legacy 모델들 (Quantum Hydra, Quantum Mamba, Gated variants)
  - `QUANTUM_HYDRA_GUIDE.md`: Quantum Hydra 모델 가이드
  - `QUANTUM_MAMBA_GUIDE.md`: Quantum Mamba 모델 가이드
  - `QUANTUM_GATED_RECURRENCE_GUIDE.md`: Gated 모델 가이드 (Legacy 확장)
  - `FOUR_MODEL_ARCHITECTURE_COMPARISON.md`: 4개 모델 비교
- `ablation/`: Ablation Study 모델들
  - `QUANTUM_SSM_README.md`: QuantumMambaSSM, QuantumHydraSSM 상세 설명
- `shared/`: 공통 아키텍처 논의
  - `QUANTUM_RECURRENCE_DISCUSSION.md`: 양자 순환 구조 논의
  - `QUANTUM_SEQUENCE_PROCESSING.md`: 양자 시퀀스 처리 이론

**사용 시나리오**:
- 모델 구조 이해
- 구현 세부사항 확인
- 아키텍처 선택 가이드

---

### 3. `02_experimental_plans/` - 실험 계획 및 프로토콜
**목적**: 실험 설계, 프로토콜, 실행 가이드

**구조**:
- `legacy/`: Legacy 실험 계획
  - `QuantumHydra_ExperimentPlan_README.md`: Quantum Hydra 실험 계획
- `ablation/`: Ablation Study 실험 계획
  - `ABLATION_STUDY_PLAN_V3.md`: **최신** Ablation Study 계획 (2×2×3 factorial design)
  - `EXPERIMENT_GUIDE.md`: 실험 실행 가이드 ⭐ **START HERE**
  - `SYNTHETIC_BENCHMARK_PROTOCOL.md`: Synthetic benchmark 프로토콜
  - `Quantum_Advantage_Test_Plan.md`: 양자 이점 테스트 계획
  - `legacy/`: 이전 버전들
    - `ABLATION_STUDY_IMPLEMENTATION_PLAN.md`: 이전 버전
    - `ABLATION_STUDY_IMPLEMENTATION_PLAN_V2.md`: 이전 버전
- `EXPERIMENTAL_PLAN_README.md`: 일반 실험 계획 (루트에 유지)

**사용 시나리오**:
- 실험 설계 이해
- 실험 실행 방법 학습
- 프로토콜 확인

---

### 4. `03_results/` - 실험 결과
**목적**: 실험 결과 요약, 분석, 성능 비교

**구조**:
- `legacy/`: Legacy 결과 (Gated 포함)
  - `QuantumHydraMamba_COMPREHENSIVE_RESULTS_2025-11-17.md`: 종합 결과 (2025-11-17)
  - `EEG_EXPERIMENT_RESULTS_README.md`: EEG 실험 결과 상세
  - `DNA_CLASSIFICATION_RESULTS.md`: DNA 분류 결과
  - `8QUBIT_RESULTS_ANALYSIS.md`: 8-qubit 결과 분석
  - `QUANTUM_GATED_RESULTS_SUMMARY.md`: Gated 모델 결과
- `ablation/`: Ablation Study 결과
  - `ABLATION_EEG_RESULTS_SUMMARY.md`: EEG Ablation 결과 요약
  - `GROUP2_PERFORMANCE_ANALYSIS.md`: Group 2 성능 분석

**사용 시나리오**:
- 실험 결과 확인
- 성능 비교
- 논문 작성 시 참조

---

### 5. `04_datasets/` - 데이터셋 가이드
**목적**: 데이터셋 설정, 사용법, 이론적 배경

**파일**:
- `EEG_DATASETS_SETUP_GUIDE.md`: EEG 데이터셋 설정 가이드
- `Forrelation_Dataset_Usage_Guide.md`: Forrelation 데이터셋 사용법
- `FORRELATION_README.md`: Forrelation 실험 가이드
- `Forrelation_Experiment_Rationale.md`: Forrelation 이론적 배경

**사용 시나리오**:
- 데이터셋 다운로드 및 설정
- 데이터 로딩 방법
- 데이터셋 특성 이해

---

### 6. `05_status_reports/` - 상태 보고서
**목적**: 진행 상황, 완료 작업, 작업 제출 현황

**파일**:
- `CURRENT_STATUS_REPORT.md`: 현재 상태 보고서
- `EXPERIMENT_STATUS_REPORT.md`: 실험 상태 보고서
- `WORK_COMPLETED_SUMMARY.md`: 완료 작업 요약
- `FIXES_COMPLETED_README.md`: 완료된 수정사항
- `JOBS_SUBMITTED_SUMMARY.md`: 제출된 작업 요약
- `TIER1_EXPERIMENTS_SUMMARY.md`: Tier 1 실험 요약

**사용 시나리오**:
- 진행 상황 확인
- 완료된 작업 추적
- 작업 제출 현황 확인

**참고**: 시간이 지나면 일부는 레거시로 이동 가능

---

### 7. `06_analysis/` - 심층 분석
**목적**: 연구 질문 답변, 복잡도 분석, 이론적 논의

**구조**:
- `legacy/`: Legacy 분석
  - `QUANTUM_GATED_COMPLEXITY_ANALYSIS.md`: Gated 모델 복잡도 분석
- `shared/`: 공통 분석
  - `RESEARCH_QUESTIONS_ANSWERS.md`: 핵심 연구 질문에 대한 답변 ⭐
  - `COMPUTATIONAL_COMPLEXITY_COMPARISON.md`: 계산 복잡도 비교
  - `MULTIGPU_QUANTUM_PARADOX.md`: Multi-GPU 양자 역설 논의
  - `TIMING_AND_METRICS.md`: 타이밍 및 메트릭 가이드

**사용 시나리오**:
- 연구 질문에 대한 답변 확인
- 복잡도 분석
- 이론적 논의 이해

---

### 8. `07_benchmarks/` - 벤치마크
**목적**: 특정 벤치마크 실험 가이드

**파일**:
- `GLUE_README.md`: GLUE 벤치마크 가이드
- `EXPERIMENTS_README.md`: 실험 개요

**사용 시나리오**:
- 벤치마크 실험 실행
- 벤치마크 결과 확인

---

### 9. `08_reference/` - 참조 및 레거시
**목적**: 레거시 문서, 빠른 참조, 설정 요약

**파일**:
- `OLD_REPO_README.md`: 이전 저장소 README
- `QUICK_REFERENCE.md`: 빠른 참조 가이드
- `UPDATED_CONFIGURATION_SUMMARY.md`: 업데이트된 설정 요약
- `OPTION_B_COMPLETE_SUMMARY.md`: Option B 완료 요약

**사용 시나리오**:
- 빠른 참조
- 레거시 정보 확인

---

## 파일 분류 상세

### Architecture Guides
```
01_architecture/
├── legacy/                      # Legacy + Gated (4개)
│   ├── QUANTUM_HYDRA_GUIDE.md
│   ├── QUANTUM_MAMBA_GUIDE.md
│   ├── QUANTUM_GATED_RECURRENCE_GUIDE.md
│   └── FOUR_MODEL_ARCHITECTURE_COMPARISON.md
├── ablation/                    # Ablation Study (1개)
│   └── QUANTUM_SSM_README.md
└── shared/                      # 공통 (2개)
    ├── QUANTUM_RECURRENCE_DISCUSSION.md
    └── QUANTUM_SEQUENCE_PROCESSING.md
```

### Experimental Plans
```
02_experimental_plans/
├── legacy/                      # Legacy (1개)
│   └── QuantumHydra_ExperimentPlan_README.md
├── ablation/                    # Ablation Study (4개)
│   ├── ABLATION_STUDY_PLAN_V3.md                    # ⭐ 최신
│   ├── EXPERIMENT_GUIDE.md                          # ⭐ START HERE
│   ├── SYNTHETIC_BENCHMARK_PROTOCOL.md
│   ├── Quantum_Advantage_Test_Plan.md
│   └── legacy/                  # 이전 버전들 (2개)
│       ├── ABLATION_STUDY_IMPLEMENTATION_PLAN.md
│       └── ABLATION_STUDY_IMPLEMENTATION_PLAN_V2.md
└── EXPERIMENTAL_PLAN_README.md  # 일반 (루트에 유지)
```

### Results
```
03_results/
├── legacy/                      # Legacy + Gated (5개)
│   ├── QuantumHydraMamba_COMPREHENSIVE_RESULTS_2025-11-17.md
│   ├── EEG_EXPERIMENT_RESULTS_README.md
│   ├── DNA_CLASSIFICATION_RESULTS.md
│   ├── 8QUBIT_RESULTS_ANALYSIS.md
│   └── QUANTUM_GATED_RESULTS_SUMMARY.md
└── ablation/                    # Ablation Study (2개)
    ├── ABLATION_EEG_RESULTS_SUMMARY.md
    └── GROUP2_PERFORMANCE_ANALYSIS.md
```

### Dataset Guides (4개)
```
04_datasets/
├── EEG_DATASETS_SETUP_GUIDE.md
├── Forrelation_Dataset_Usage_Guide.md
├── FORRELATION_README.md
└── Forrelation_Experiment_Rationale.md
```

### Status Reports (6개)
```
05_status_reports/
├── CURRENT_STATUS_REPORT.md
├── EXPERIMENT_STATUS_REPORT.md
├── WORK_COMPLETED_SUMMARY.md
├── FIXES_COMPLETED_README.md
├── JOBS_SUBMITTED_SUMMARY.md
└── TIER1_EXPERIMENTS_SUMMARY.md
```

### Analysis
```
06_analysis/
├── legacy/                      # Legacy (1개)
│   └── QUANTUM_GATED_COMPLEXITY_ANALYSIS.md
└── shared/                      # 공통 (4개)
    ├── RESEARCH_QUESTIONS_ANSWERS.md                 # ⭐ 핵심
    ├── COMPUTATIONAL_COMPLEXITY_COMPARISON.md
    ├── MULTIGPU_QUANTUM_PARADOX.md
    └── TIMING_AND_METRICS.md
```

### Benchmarks (2개)
```
07_benchmarks/
├── GLUE_README.md
└── EXPERIMENTS_README.md
```

### Reference (4개)
```
08_reference/
├── OLD_REPO_README.md
├── QUICK_REFERENCE.md
├── UPDATED_CONFIGURATION_SUMMARY.md
└── OPTION_B_COMPLETE_SUMMARY.md
```

---

## 장점

### 1. 명확한 카테고리 구분
- 문서의 목적이 명확함
- 찾고자 하는 문서를 빠르게 발견 가능

### 2. 확장성
- 새로운 문서 추가 시 적절한 카테고리에 배치
- 카테고리별로 독립적으로 관리 가능

### 3. 사용자 경험 개선
- 신규 사용자: `00_index/README.md` → `02_experimental_plans/EXPERIMENT_GUIDE.md`
- 연구자: `06_analysis/RESEARCH_QUESTIONS_ANSWERS.md`
- 개발자: `01_architecture/` 시리즈

### 4. 유지보수 용이
- 레거시 문서는 `08_reference/`로 분리
- 최신 문서는 명확히 표시 (V3, 최신 날짜 등)

---

## 구현 상태

✅ **리팩토링 완료** (2026년 1월)

파일들이 Legacy/Ablation/Shared로 명시적으로 분리되었습니다:
- Legacy와 Gated를 `legacy/`에 통합
- Ablation Study 관련 문서를 `ablation/`로 분리
- 공통 문서는 `shared/`로 분리

---

## 메인 인덱스

`00_index/README.md`는 Legacy/Ablation/Shared 구조를 반영하여 업데이트되었습니다.

---

## 리팩토링 완료 상태

1. ✅ 카테고리 구조 확정 (Legacy/Ablation/Shared)
2. ✅ 파일 이동 완료
3. ✅ 메인 인덱스 업데이트 완료
4. ⚠️ 문서 내부 링크 업데이트 필요 (각 문서의 상대 경로 참조 수정)
