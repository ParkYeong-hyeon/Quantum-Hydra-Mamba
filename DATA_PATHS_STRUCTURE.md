# 데이터 경로 구조 (Data Loader Expected Paths)

각 데이터 로더가 기대하는 데이터 경로 구조입니다.

## 📁 전체 경로 구조

```
프로젝트 루트: /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba/
│
├── 📂 data/                                    # 프로젝트 루트 기준 상대 경로
│   ├── 📂 synthetic_benchmarks/               # Synthetic 데이터셋
│   │   ├── 📂 forrelation/
│   │   │   ├── forrelation_L50_seed2024.pt
│   │   │   ├── forrelation_L100_seed2024.pt
│   │   │   ├── forrelation_L200_seed2024.pt
│   │   │   └── ... (다른 seq_len, seed 조합)
│   │   ├── 📂 adding_problem/
│   │   │   ├── adding_L100_seed2024.pt
│   │   │   ├── adding_L200_seed2024.pt
│   │   │   ├── adding_L500_seed2024.pt
│   │   │   ├── adding_L1000_seed2024.pt
│   │   │   └── ... (다른 seq_len, seed 조합)
│   │   └── 📂 selective_copy/
│   │       ├── selective_copy_L100_M8_seed2024.pt
│   │       ├── selective_copy_L200_M8_seed2024.pt
│   │       ├── selective_copy_L500_M8_seed2024.pt
│   │       ├── selective_copy_L1000_M8_seed2024.pt
│   │       └── ... (다른 seq_len, num_markers, seed 조합)
│   │
│   └── 📂 dna/                                 # DNA Promoter 데이터셋
│       └── promoters.data                      # UCI에서 다운로드 또는 자동 생성
│
├── 📂 SEED/                                    # SEED EEG 데이터셋 (선택적)
│   └── 📂 SEED_EEG/
│       └── 📂 Preprocessed_EEG/                # 또는 ./Preprocessed_EEG (루트 기준)
│           └── ... (TorchEEG가 기대하는 구조)
│
├── 📂 Processed_data/                          # FACED EEG 데이터셋 (선택적)
│   ├── 📂 Processed_data/                      # 실제 데이터 폴더
│   │   └── ... (TorchEEG가 기대하는 구조)
│   └── 📂 faced_io/                            # TorchEEG 캐시
│
└── 📂 Preprocessed_EEG/                        # SEED EEG 대체 경로 (선택적)
    └── ... (TorchEEG가 기대하는 구조)

─────────────────────────────────────────────────────────────────────────────

절대 경로 (하드코딩):
└── /pscratch/sd/j/junghoon/PhysioNet_EEG/     # PhysioNet EEG 데이터셋
    └── ... (MNE가 기대하는 구조)
        ├── S001/
        │   ├── S001R04.edf
        │   ├── S001R08.edf
        │   └── S001R12.edf
        ├── S002/
        └── ... (S001 ~ S109)

─────────────────────────────────────────────────────────────────────────────

홈 디렉토리 기준:
└── ~/.genomic_benchmarks/                      # Genomic Benchmarks 캐시
    ├── human_nontata_promoters/
    │   ├── train/
    │   │   ├── 0/                              # 클래스 0
    │   │   │   ├── seq_001.txt
    │   │   │   └── ...
    │   │   └── 1/                              # 클래스 1
    │   │       ├── seq_001.txt
    │   │       └── ...
    │   └── test/
    │       ├── 0/
    │       └── 1/
    ├── human_enhancers_cohn/
    ├── demo_human_or_worm/
    └── ... (다른 genomic benchmark 데이터셋)
```

## 📋 데이터 로더별 상세 경로

### 1. **PhysioNet EEG** (`Load_PhysioNet_EEG_NoPrompt.py`)
- **경로**: `/pscratch/sd/j/junghoon/PhysioNet_EEG` (하드코딩, 절대 경로)
- **사용 스크립트**: 
  - `run_ablation_eeg.py`
  - `run_single_model_eeg.py`
  - `run_gated_models.py` (dataset='eeg')
- **구조**: MNE 패키지가 자동으로 다운로드/관리
- **참고**: `mne.datasets.eegbci.load_data()` 사용

### 2. **Synthetic Benchmarks** (`run_synthetic_benchmark.py`)
- **기본 경로**: `./data/synthetic_benchmarks/` (프로젝트 루트 기준)
- **하위 구조**:
  ```
  ./data/synthetic_benchmarks/
  ├── forrelation/
  │   └── forrelation_L{seq_len}_seed{seed}.pt
  ├── adding_problem/
  │   └── adding_L{seq_len}_seed{seed}.pt
  └── selective_copy/
      └── selective_copy_L{seq_len}_M{num_markers}_seed{seed}.pt
  ```
- **자동 생성**: 파일이 없으면 자동으로 생성
- **생성 스크립트**: `scripts/generate_all_synthetic_datasets.py`

### 3. **DNA Promoter** (`Load_DNA_Sequences.py`)
- **경로**: `./data/dna/promoters.data` (프로젝트 루트 기준)
- **다운로드 URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data
- **자동 생성**: 다운로드 실패 시 synthetic 데이터 자동 생성
- **사용 스크립트**:
  - `run_single_model_dna.py`
  - `run_gated_models.py` (dataset='dna')

### 4. **Genomic Benchmarks** (`Load_Genomic_Benchmarks.py`)
- **기본 경로**: `~/.genomic_benchmarks/` (홈 디렉토리 기준)
- **구조**:
  ```
  ~/.genomic_benchmarks/
  └── {dataset_name}/
      ├── train/
      │   ├── 0/  또는 negative/  (클래스별 디렉토리)
      │   │   └── *.txt (시퀀스 파일)
      │   └── 1/  또는 positive/
      │       └── *.txt
      └── test/
          ├── 0/
          └── 1/
  ```
- **지원 데이터셋**:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`
  - `human_ocr_ensembl`
  - `demo_coding_vs_intergenomic_seqs`
  - `demo_human_or_worm`
  - `drosophila_enhancers_stark`
  - `dummy_mouse_enhancers_ensembl`
- **자동 다운로드**: `genomic_benchmarks` 패키지가 자동 처리
- **사용 스크립트**:
  - `run_ssm_genomic_comparison.py`
  - `run_full_comparison.py`
  - `run_gated_genomic.py`

### 5. **SEED EEG** (`Load_SEED_EEG.py`)
- **기본 경로**: 
  - `./SEED/SEED_EEG/Preprocessed_EEG` (우선)
  - 또는 `./Preprocessed_EEG` (대체)
- **IO 캐시**: `./SEED/seed_io` (기본값)
- **요구사항**: TorchEEG 패키지 필요
- **구조**: TorchEEG가 기대하는 구조

### 6. **FACED EEG** (`Load_FACED_EEG.py`)
- **기본 경로**: `./Processed_data/Processed_data`
- **IO 캐시**: `./Processed_data/faced_io` (기본값)
- **요구사항**: TorchEEG 패키지 필요
- **구조**: TorchEEG가 기대하는 구조
- **라벨 타입**: 
  - `valence` (3-class)
  - `emotion` (9-class)

## 🔍 경로 확인 체크리스트

### 필수 (주요 실험용)
- [ ] `/pscratch/sd/j/junghoon/PhysioNet_EEG` 존재 여부 확인
- [ ] `./data/synthetic_benchmarks/` 디렉토리 생성 가능 여부 확인
- [ ] `./data/dna/` 디렉토리 생성 가능 여부 확인

### 선택적 (특정 실험용)
- [ ] `~/.genomic_benchmarks/` 디렉토리 생성 가능 여부 확인
- [ ] SEED EEG 데이터 (필요한 경우)
- [ ] FACED EEG 데이터 (필요한 경우)

## 📝 참고사항

1. **상대 경로**: 대부분의 데이터는 프로젝트 루트(`Quantum-Hydra-Mamba/`) 기준 상대 경로 사용
2. **절대 경로**: PhysioNet EEG만 하드코딩된 절대 경로 사용
3. **자동 생성**: Synthetic, DNA 데이터는 없으면 자동 생성
4. **자동 다운로드**: Genomic Benchmarks는 `genomic_benchmarks` 패키지가 자동 다운로드
5. **환경 변수**: 일부 데이터 로더는 `root_path` 파라미터로 경로 변경 가능
