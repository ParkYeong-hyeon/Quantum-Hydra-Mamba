"""
Job 실행 환경 설정 (인프라 관련)
환경 변수로 오버라이드 가능

사용법:
    1. 환경 변수 설정 (권장):
       export QHM_PYTHON_PATH=/scratch/connectome/mandy/envs/qhydra/bin/python
       export QHM_CONDA_ENV=qhydra
       export QHM_CONDA_ENV_PATH=/path/to/conda/env
       export QHM_SLURM_ACCOUNT=m4727_g
       export QHM_SLURM_CONSTRAINT=gpu&hbm80g
       export QHM_SLURM_QOS=shared
       export QHM_SLURM_TIME=24:00:00
       export QHM_SLURM_NODES=1
       export QHM_SLURM_GPUS=1
       export QHM_SLURM_CPUS=32
    
    2. 또는 아래 기본값을 직접 수정
"""
import os
from pathlib import Path
from .data_paths import PROJECT_ROOT  # 프로젝트 루트 재사용

# ============================================
# 파이썬 경로 설정
# ============================================
# 환경 변수로 오버라이드 가능, 없으면 기본값 사용
PY = os.getenv('QHM_PYTHON_PATH', '/scratch/connectome/mandy/envs/qhydra/bin/python')

# ============================================
# Conda/Micromamba 환경 설정
# ============================================
CONDA_ENV = os.getenv('QHM_CONDA_ENV', 'qhydra')
CONDA_ENV_PATH = os.getenv('QHM_CONDA_ENV_PATH', '')
MICROMAMBA_PATH = Path(os.getenv(
    'QHM_MICROMAMBA_PATH',
    '/home/connectome/mandy/.local/bin/micromamba'
))

# ============================================
# SLURM 기본 설정
# ============================================
SLURM_DEFAULTS = {
    'account': os.getenv('QHM_SLURM_ACCOUNT', 'm4727_g'),
    'constraint': os.getenv('QHM_SLURM_CONSTRAINT', 'gpu&hbm80g'),
    'qos': os.getenv('QHM_SLURM_QOS', 'shared'),
    'time': os.getenv('QHM_SLURM_TIME', '24:00:00'),
    'nodes': int(os.getenv('QHM_SLURM_NODES', '1')),
    'gpus': int(os.getenv('QHM_SLURM_GPUS', '1')),
    'cpus_per_task': int(os.getenv('QHM_SLURM_CPUS', '32')),
}

# ============================================
# 경로 헬퍼 함수
# ============================================
def get_output_dir(experiment_name: str) -> Path:
    """
    실험 결과 저장 디렉토리 가져오기
    
    Args:
        experiment_name: 실험 이름 (예: 'ablation_eeg', 'dna', 'eeg', etc.)
    
    Returns:
        Path 객체 (PROJECT_ROOT / 'results' / experiment_name)
    """
    return PROJECT_ROOT / 'results' / experiment_name

def get_jobs_dir(experiment_name: str) -> Path:
    """
    Job 스크립트 저장 디렉토리 가져오기
    
    Args:
        experiment_name: 실험 이름 (예: 'ablation_eeg', 'dna', 'eeg', etc.)
    
    Returns:
        Path 객체 (PROJECT_ROOT / 'jobs' / experiment_name)
    """
    return PROJECT_ROOT / 'jobs' / experiment_name

def get_script_path(script_name: str) -> Path:
    """
    Training 스크립트 경로 가져오기
    
    Args:
        script_name: 스크립트 이름 (예: 'run_ablation_eeg.py', 'run_single_model_dna.py')
    
    Returns:
        Path 객체 (PROJECT_ROOT / 'scripts' / 'training' / script_name)
    """
    return PROJECT_ROOT / 'scripts' / 'training' / script_name

def get_slurm_config(**overrides) -> dict:
    """
    SLURM 설정 가져오기 (오버라이드 가능)
    
    Args:
        **overrides: 기본값을 오버라이드할 설정들
    
    Returns:
        SLURM 설정 딕셔너리
    """
    config = SLURM_DEFAULTS.copy()
    config.update(overrides)
    return config
