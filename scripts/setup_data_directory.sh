#!/bin/bash
# Setup data directory structure with symbolic links
# Usage: bash scripts/setup_data_directory.sh
#
# Data root must be set via:
#   - Environment variable: QHM_DATA_ROOT
#   - Or edit config/data_paths.py: DATA_ROOT variable

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python 설정 파일에서 DATA_ROOT 읽기
if [ -z "$QHM_DATA_ROOT" ]; then
    # config/data_paths.py에서 DATA_ROOT 추출
    DATA_ROOT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from config.data_paths import DATA_ROOT
print(DATA_ROOT)
" 2>/dev/null || echo "")
    
    if [ -z "$DATA_ROOT" ]; then
        echo "❌ Error: QHM_DATA_ROOT environment variable not set"
        echo ""
        echo "Please set it in one of these ways:"
        echo "  1. Environment variable:"
        echo "     export QHM_DATA_ROOT=/path/to/your/data"
        echo ""
        echo "  2. Edit config/data_paths.py:"
        echo "     Change the DATA_ROOT variable"
        echo ""
        exit 1
    fi
else
    DATA_ROOT="$QHM_DATA_ROOT"
fi

echo "=========================================="
echo "Setting up data directory structure"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Project root: $PROJECT_ROOT"
echo ""

# 1. 공통 데이터 디렉토리 생성
echo "Creating data root directory..."
mkdir -p "$DATA_ROOT"/{synthetic_benchmarks,dna,SEED,Processed_data,physionet,genomic_benchmarks}

# 2. 프로젝트 내부 심볼릭 링크 생성
echo "Creating symbolic links..."
cd "$PROJECT_ROOT"

# 기존 data가 일반 디렉토리면 백업
if [ -d "data" ] && [ ! -L "data" ]; then
    BACKUP_NAME="data.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing data directory to $BACKUP_NAME..."
    mv data "$BACKUP_NAME"
    echo "⚠️  Note: If you had data in the old directory, move it to: $DATA_ROOT"
fi

# 심볼릭 링크 생성
if [ ! -L "data" ]; then
    ln -s "$DATA_ROOT" data
    echo "✅ Created: data -> $DATA_ROOT"
else
    echo "ℹ️  Symbolic link already exists: data"
fi

# 3. 검증
echo ""
echo "Verification:"
if [ -L "data" ]; then
    LINK_TARGET=$(readlink -f data)
    echo "✅ data -> $LINK_TARGET"
    if [ "$LINK_TARGET" = "$DATA_ROOT" ]; then
        echo "✅ Link target matches data root"
    else
        echo "⚠️  Warning: Link target doesn't match data root"
        echo "   Expected: $DATA_ROOT"
        echo "   Actual: $LINK_TARGET"
    fi
else
    echo "❌ Error: data is not a symbolic link"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Data root: $DATA_ROOT"
echo ""
echo "To change the data root, edit config/data_paths.py:"
echo "  Change the DATA_ROOT variable in config/data_paths.py"
echo ""
echo "Or set environment variable:"
echo "  export QHM_DATA_ROOT=/path/to/your/data"
echo "  bash scripts/setup_data_directory.sh"
