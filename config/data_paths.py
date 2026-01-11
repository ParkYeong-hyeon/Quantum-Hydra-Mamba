"""
데이터 경로 설정
환경 변수로 오버라이드 가능
"""
import os
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent

# 공통 데이터 루트
# 설정 방법:
#   1. 환경 변수 설정: export QHM_DATA_ROOT=/path/to/your/data
#   2. 또는 아래 기본값을 직접 수정
DATA_ROOT = Path(os.getenv(
    'QHM_DATA_ROOT',
    '/scratch/connectome/mandy/projects/data/Quantum-Hydra-Mamba'  # 여기를 수정하세요
))

# 프로젝트 내부 데이터 경로 (심볼릭 링크를 통해 실제 데이터에 접근)
DATA_DIRS = {
    'synthetic': PROJECT_ROOT / 'data' / 'synthetic_benchmarks',
    'dna': PROJECT_ROOT / 'data' / 'dna',
    'seed': PROJECT_ROOT / 'SEED',
    'faced': PROJECT_ROOT / 'Processed_data',
    'physionet': PROJECT_ROOT / 'data' / 'physionet',  # PhysioNet EEG 데이터
    'genomic': PROJECT_ROOT / 'data' / 'genomic_benchmarks',  # Genomic Benchmarks 캐시
}

# 절대 경로 데이터 (현재 사용되지 않음, DATA_DIRS로 이동됨)
# 참고: physionet과 genomic은 이제 DATA_DIRS에 포함되어 프로젝트 내부 경로 사용
ABSOLUTE_DATA_PATHS = {
    # 필요시 다른 절대 경로 데이터 추가 가능
}

def get_data_path(key: str, absolute: bool = False) -> Path:
    """
    데이터 경로 가져오기
    
    Args:
        key: 데이터셋 키 ('synthetic', 'dna', 'physionet', etc.)
        absolute: True면 절대 경로 반환, False면 프로젝트 기준 상대 경로
    
    Returns:
        Path 객체
    
    Raises:
        ValueError: 알 수 없는 키인 경우
    """
    if absolute:
        if key not in ABSOLUTE_DATA_PATHS:
            raise ValueError(
                f"Unknown absolute data key: {key}. "
                f"Available: {list(ABSOLUTE_DATA_PATHS.keys())}"
            )
        return ABSOLUTE_DATA_PATHS[key]
    else:
        if key not in DATA_DIRS:
            raise ValueError(
                f"Unknown data key: {key}. "
                f"Available: {list(DATA_DIRS.keys())}"
            )
        return DATA_DIRS[key]

# 편의 함수
def get_synthetic_path() -> Path:
    """Synthetic benchmark 데이터 경로"""
    return get_data_path('synthetic')

def get_dna_path() -> Path:
    """DNA 데이터 경로"""
    return get_data_path('dna')

def get_physionet_path() -> Path:
    """PhysioNet EEG 데이터 경로"""
    return get_data_path('physionet')

def get_genomic_path() -> Path:
    """Genomic benchmarks 데이터 경로"""
    return get_data_path('genomic')
