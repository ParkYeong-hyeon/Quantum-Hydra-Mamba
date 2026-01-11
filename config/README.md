# Configuration Module

데이터 경로 및 프로젝트 설정을 관리하는 모듈입니다.

## 경로 설정 방법

### 방법 1: 설정 파일 직접 수정 (권장)

`config/data_paths.py` 파일을 열어서 직접 수정:

```python
# config/data_paths.py
DATA_ROOT = Path('/your/custom/data/path')  # 여기 수정

# DATA_DIRS에 모든 데이터 경로 포함
DATA_DIRS = {
    'synthetic': PROJECT_ROOT / 'data' / 'synthetic_benchmarks',
    'dna': PROJECT_ROOT / 'data' / 'dna',
    'physionet': PROJECT_ROOT / 'data' / 'physionet',  # 여기 수정
    'genomic': PROJECT_ROOT / 'data' / 'genomic_benchmarks',  # 여기 수정
    # ...
}
```

### 방법 2: 환경 변수 사용

```bash
# 데이터 루트 변경
export QHM_DATA_ROOT=/path/to/your/data

# PhysioNet EEG 경로 변경
export PHYSIONET_EEG_PATH=/path/to/physionet

# Genomic Benchmarks 경로 변경
export GENOMIC_BENCHMARKS_PATH=/path/to/genomic
```

## 사용 방법

```python
from config.data_paths import get_data_path, get_synthetic_path, get_physionet_path

# Synthetic 데이터 경로
synthetic_path = get_synthetic_path()
# 또는
synthetic_path = get_data_path('synthetic')

# PhysioNet EEG 경로
physionet_path = get_physionet_path()
# 또는
physionet_path = get_data_path('physionet')

# Genomic Benchmarks 경로
genomic_path = get_genomic_path()
# 또는
genomic_path = get_data_path('genomic')
```

## 데이터 경로 구조

### 프로젝트 내부 경로 (심볼릭 링크)

- `synthetic`: `./data/synthetic_benchmarks/`
- `dna`: `./data/dna/`
- `seed`: `./SEED/`
- `faced`: `./Processed_data/`
- `physionet`: `./data/physionet/` (PhysioNet EEG 데이터)
- `genomic`: `./data/genomic_benchmarks/` (Genomic Benchmarks 캐시)

### 절대 경로

- 현재 사용되지 않음 (모든 데이터가 프로젝트 내부 경로 사용)

## 심볼릭 링크 설정

데이터 디렉토리 구조를 설정하려면:

```bash
bash scripts/setup_data_directory.sh
```

이 스크립트는:
1. 공통 데이터 디렉토리 생성 (`/scratch/connectome/mandy/projects/data/Quantum-Hydra-Mamba`)
2. 프로젝트 내부에 심볼릭 링크 생성 (`data -> /scratch/.../data/Quantum-Hydra-Mamba`)

## 참고

- 심볼릭 링크를 통해 프로젝트 내부의 상대 경로(`./data/`)가 실제 데이터 저장소를 가리킵니다.
- 환경 변수 `QHM_DATA_ROOT`로 데이터 저장 위치를 변경할 수 있습니다.
- 기존 코드는 수정 없이 작동합니다 (상대 경로 유지).
