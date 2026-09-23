#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script instead of executing it:" >&2
    echo "  source scripts/activate_env.sh" >&2
    exit 2
fi

ELOGGEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ELOGGEN_PROJECT_ROOT/.eloggen.local" ]]; then
    source "$ELOGGEN_PROJECT_ROOT/.eloggen.local"
fi
ELOGGEN_DATA_ROOT="${ELOGGEN_DATA_ROOT:-$ELOGGEN_PROJECT_ROOT}"
ELOGGEN_THIRD_PARTY_ROOT="${ELOGGEN_THIRD_PARTY_ROOT:-$ELOGGEN_DATA_ROOT/third_party}"
ELOGGEN_BEHAVIOR_ROOT="${ELOGGEN_BEHAVIOR_ROOT:-$ELOGGEN_THIRD_PARTY_ROOT/BEHAVIOR-1K}"
ELOGGEN_ROBOMIMIC_ROOT="${ELOGGEN_ROBOMIMIC_ROOT:-$ELOGGEN_THIRD_PARTY_ROOT/robomimic}"
ELOGGEN_DATASET_ROOT="${ELOGGEN_DATASET_ROOT:-$ELOGGEN_BEHAVIOR_ROOT/datasets}"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
    echo "Activate the ElogGen Conda environment before sourcing this script." >&2
    echo "For example: conda activate eloggen" >&2
    return 1
fi
ELOGGEN_ENV_PREFIX="$(cd "$CONDA_PREFIX" && pwd)"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "ElogGen requires Python 3.10; got " + sys.version' || return 1

if [[ ! -d "$ELOGGEN_DATASET_ROOT" ]]; then
    echo "BEHAVIOR datasets are not available at $ELOGGEN_DATASET_ROOT" >&2
    echo "Run bash setup.sh first." >&2
    return 1
fi

if [[ ! -f "$ELOGGEN_ENV_PREFIX/conda-meta/history" ]]; then
    echo "ElogGen environment is not installed at $ELOGGEN_ENV_PREFIX" >&2
    return 1
fi
if [[ ! -d "$ELOGGEN_BEHAVIOR_ROOT/OmniGibson" ]]; then
    echo "BEHAVIOR-1K checkout is not available at $ELOGGEN_BEHAVIOR_ROOT" >&2
    return 1
fi
if [[ ! -d "$ELOGGEN_ROBOMIMIC_ROOT/robomimic" ]]; then
    echo "robomimic checkout is not available at $ELOGGEN_ROBOMIMIC_ROOT" >&2
    return 1
fi

export PIP_CACHE_DIR="$ELOGGEN_DATA_ROOT/cache/pip"
export XDG_CACHE_HOME="$ELOGGEN_DATA_ROOT/cache"
export ELOGGEN_ROOT="$ELOGGEN_PROJECT_ROOT"
export ELOGGEN_DATA_ROOT
export ELOGGEN_THIRD_PARTY_ROOT
export ELOGGEN_ENV_PREFIX
export ELOGGEN_BEHAVIOR_ROOT
export ELOGGEN_ROBOMIMIC_ROOT
export ELOGGEN_DATASET_ROOT
export PYTHONPATH="$ELOGGEN_PROJECT_ROOT/src:$ELOGGEN_BEHAVIOR_ROOT/OmniGibson:$ELOGGEN_ROBOMIMIC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$ELOGGEN_ENV_PREFIX"
export CUDACXX="$ELOGGEN_ENV_PREFIX/bin/nvcc"
export CC="$ELOGGEN_ENV_PREFIX/bin/gcc"
export CXX="$ELOGGEN_ENV_PREFIX/bin/g++"
ELOGGEN_CUDA_TARGET="$ELOGGEN_ENV_PREFIX/targets/x86_64-linux"
export CPATH="$ELOGGEN_CUDA_TARGET/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$ELOGGEN_CUDA_TARGET/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$ELOGGEN_CUDA_TARGET/lib:$ELOGGEN_ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ELOGGEN_CUDA_TARGET/lib:$ELOGGEN_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "ElogGen environment: $ELOGGEN_ENV_PREFIX"
echo "Third-party root: $ELOGGEN_THIRD_PARTY_ROOT"
echo "BEHAVIOR-1K source: $ELOGGEN_BEHAVIOR_ROOT"
echo "robomimic source: $ELOGGEN_ROBOMIMIC_ROOT"
echo "BEHAVIOR datasets: $ELOGGEN_DATASET_ROOT"
