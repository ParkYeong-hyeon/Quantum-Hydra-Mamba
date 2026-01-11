# 데이터 Git 관리 전략 (Data Git Management Strategy)

## 📋 일반적인 베스트 프랙티스

### ❌ Git 레포 안에 두지 않는 것들

1. **큰 데이터 파일** (수 GB 이상)
   - Git은 대용량 파일 관리에 비효율적
   - 레포 크기 증가, clone 시간 증가
   - 예: 원본 EEG 데이터, 이미지 데이터셋

2. **자동 생성되는 데이터**
   - 코드로 재생성 가능한 데이터
   - 예: Synthetic 데이터, 전처리된 데이터

3. **환경별/사용자별 경로의 데이터**
   - 절대 경로가 하드코딩된 데이터
   - 예: `/pscratch/sd/j/junghoon/PhysioNet_EEG`

4. **캐시 파일**
   - 임시 파일, 중간 결과물
   - 예: `*.pt` (PyTorch 체크포인트), `__pycache__/`

### ✅ Git 레포 안에 둘 수 있는 것들

1. **작은 샘플 데이터** (수 MB 이하)
   - 예시/테스트용 작은 데이터셋
   - 예: `data/samples/`, `data/examples/`

2. **설정 파일 및 메타데이터**
   - 데이터셋 설명, 스키마 정의
   - 예: `data/README.md`, `data/schema.json`

3. **데이터 생성 스크립트**
   - 데이터를 재생성하는 코드
   - 예: `scripts/generate_all_synthetic_datasets.py`

---

## 🔍 현재 프로젝트 분석

### 현재 `.gitignore` 설정

```gitignore
# Large data directories
faced_io_test/
Processed_data/
data/              # ✅ 이미 무시됨
results/           # ✅ 결과물도 무시됨
```

### 프로젝트 데이터 분류

| 데이터 타입 | 경로 | 크기 추정 | Git 관리 | 이유 |
|-----------|------|----------|---------|------|
| **PhysioNet EEG** | `/pscratch/.../PhysioNet_EEG` | ~수 GB | ❌ 밖에 | 절대 경로, 하드코딩 |
| **Synthetic** | `./data/synthetic_benchmarks/` | ~수 MB-수 GB | ❌ 밖에 | 자동 생성 가능 |
| **DNA** | `./data/dna/promoters.data` | ~수 KB | ⚠️ 선택적 | 작지만 자동 다운로드 |
| **Genomic** | `~/.genomic_benchmarks/` | ~수 GB | ❌ 밖에 | 홈 디렉토리, 자동 다운로드 |
| **SEED EEG** | `./SEED/.../Preprocessed_EEG/` | ~수 GB | ❌ 밖에 | 큰 파일 |
| **FACED EEG** | `./Processed_data/` | ~8 GB | ❌ 밖에 | 매우 큰 파일 |

---

## ✅ 권장 전략

### 1. **현재 설정 유지** (권장)

현재 `.gitignore`가 이미 적절하게 설정되어 있습니다:

```gitignore
data/              # 모든 데이터 디렉토리 무시
results/           # 결과물 무시
Processed_data/    # FACED EEG 무시
```

**장점:**
- ✅ 자동 생성 데이터가 Git에 포함되지 않음
- ✅ 큰 파일이 레포를 무겁게 만들지 않음
- ✅ 환경별 경로 문제 없음

### 2. **추가 고려사항**

#### A. 작은 샘플 데이터는 Git에 포함 가능

만약 작은 예시 데이터가 필요하다면:

```bash
# .gitignore에 예외 추가
!data/samples/     # 작은 샘플 데이터는 포함
!data/README.md    # 데이터 설명 문서는 포함
```

**예시 구조:**
```
data/
├── samples/              # ✅ Git에 포함 (작은 샘플)
│   ├── eeg_sample.pt    # ~1MB 이하
│   └── dna_sample.pt
├── synthetic_benchmarks/ # ❌ Git에서 무시 (자동 생성)
├── dna/                  # ❌ Git에서 무시 (자동 다운로드)
└── README.md             # ✅ Git에 포함 (설명 문서)
```

#### B. 데이터 경로 문서화

`DATA_PATHS_STRUCTURE.md` (이미 생성됨)를 통해:
- ✅ 데이터 위치 명시
- ✅ 다운로드/생성 방법 문서화
- ✅ Git에 포함되지 않아도 사용자가 데이터를 찾을 수 있음

#### C. 데이터 생성 스크립트는 Git에 포함

```bash
scripts/
├── generate_all_synthetic_datasets.py  # ✅ Git에 포함
└── ...
```

**이유:**
- 코드로 데이터 재생성 가능
- 다른 사용자가 동일한 데이터 생성 가능
- 재현성 보장

---

## 🎯 최종 권장사항

### ✅ **현재 설정 유지 + 선택적 개선**

1. **`.gitignore`는 그대로 유지**
   - `data/` 디렉토리 무시
   - `results/` 디렉토리 무시
   - 이미 적절한 설정

2. **선택적: 작은 샘플 데이터 추가** (선택사항)
   ```bash
   # .gitignore에 추가
   !data/samples/
   !data/README.md
   ```

3. **문서화 강화** (이미 완료)
   - `DATA_PATHS_STRUCTURE.md` ✅
   - 데이터 다운로드/생성 방법 명시

4. **데이터 생성 스크립트는 Git에 포함** (이미 포함됨)
   - `scripts/generate_all_synthetic_datasets.py` ✅

---

## 📊 다른 프로젝트와의 비교

### 일반적인 ML 프로젝트 패턴

```
project/
├── .gitignore          # data/, results/ 무시
├── data/               # ❌ Git에서 무시
│   ├── raw/            # 원본 데이터
│   ├── processed/      # 전처리된 데이터
│   └── README.md       # ✅ Git에 포함 (설명)
├── scripts/
│   └── download_data.py # ✅ Git에 포함 (다운로드 스크립트)
└── requirements.txt     # ✅ Git에 포함
```

### 현재 프로젝트 패턴 (권장)

```
Quantum-Hydra-Mamba/
├── .gitignore          # ✅ data/, results/ 무시
├── data/               # ❌ Git에서 무시
│   ├── synthetic_benchmarks/  # 자동 생성
│   └── dna/                  # 자동 다운로드
├── scripts/
│   └── generate_all_synthetic_datasets.py  # ✅ Git에 포함
├── DATA_PATHS_STRUCTURE.md    # ✅ Git에 포함 (문서)
└── docs/04_datasets/          # ✅ Git에 포함 (설명)
```

**결론: 현재 프로젝트는 일반적인 베스트 프랙티스를 따르고 있습니다! ✅**

---

## 🔧 실전 체크리스트

### Git에 포함되어야 하는 것
- [x] 데이터 생성 스크립트 (`scripts/generate_*.py`)
- [x] 데이터 경로 문서 (`DATA_PATHS_STRUCTURE.md`)
- [x] 데이터 로더 코드 (`data_loaders/`)
- [ ] (선택) 작은 샘플 데이터 (`data/samples/`)

### Git에서 무시되어야 하는 것
- [x] 자동 생성 데이터 (`data/synthetic_benchmarks/`)
- [x] 다운로드된 데이터 (`data/dna/`)
- [x] 큰 원본 데이터 (`Processed_data/`, `SEED/`)
- [x] 결과물 (`results/`)
- [x] 캐시 파일 (`*.pt`, `__pycache__/`)

---

## 💡 추가 팁

### 대용량 데이터 관리 대안

만약 데이터를 공유해야 한다면:

1. **Git LFS** (Large File Storage)
   - Git 확장 기능
   - 큰 파일을 별도 저장소에 저장
   - ⚠️ 하지만 자동 생성 데이터에는 불필요

2. **외부 저장소**
   - Google Drive, Dropbox
   - AWS S3, Google Cloud Storage
   - 데이터셋 전용 플랫폼 (Kaggle, HuggingFace)

3. **데이터 다운로드 스크립트**
   - `scripts/download_data.sh`
   - 공개 URL에서 자동 다운로드
   - ✅ 현재 프로젝트는 이미 이 방식 사용 중

---

## ✅ 결론

**현재 프로젝트의 Git 전략은 적절합니다!**

- ✅ `.gitignore`에 `data/` 포함됨
- ✅ 데이터 생성 스크립트는 Git에 포함됨
- ✅ 문서화가 잘 되어 있음
- ✅ 자동 생성/다운로드 가능한 데이터는 Git 밖에 있음

**추가 개선사항 (선택적):**
- 작은 샘플 데이터를 `data/samples/`에 추가하고 Git에 포함
- `.gitignore`에 `!data/samples/` 예외 추가
