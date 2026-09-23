#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/.eloggen.local" ]]; then
    source "$PROJECT_ROOT/.eloggen.local"
fi
DEFAULT_DATA_ROOT="$PROJECT_ROOT"
BEHAVIOR_REPOSITORY="${ELOGGEN_BEHAVIOR_REPOSITORY:-https://github.com/StanfordVL/BEHAVIOR-1K.git}"
ROBOMIMIC_REPOSITORY="${ELOGGEN_ROBOMIMIC_REPOSITORY:-https://github.com/ChengshuLi/robomimic.git}"
GIT_PROXY="${ELOGGEN_GIT_PROXY:-}"
TORCH_INDEX_URL="${ELOGGEN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CUROBO_COMMIT="d083916436f1f749cdd351bdf96c4c5df4e6d192"
CUROBO_VERSION="0.0.post1.dev87"
CUROBO_ARCHIVE_SHA256="2e34038fbb03a1ce1581938cf5804189bcf4ee12fec8852371cc840a35a2d644"
CUROBO_ARCHIVE_URL="https://codeload.github.com/StanfordVL/curobo/tar.gz/$CUROBO_COMMIT"

DATA_ROOT="${ELOGGEN_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
PREPARE_ONLY=false
DRY_RUN=false
SKIP_TESTS=false
CAPTURE_ENVIRONMENT=true

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_env.sh [options]

Options:
  --data-root PATH          Location for third-party sources, downloads and caches.
  --behavior-repository URL BEHAVIOR-1K Git URL (HTTPS, SSH, or mirror).
  --robomimic-repository URL
                            robomimic Git URL (HTTPS, SSH, or mirror).
  --git-proxy URL           HTTP(S) proxy used for Git source fetches.
  --prepare-only            Clone and patch sources; do not install packages.
  --skip-tests              Skip ElogGen unit tests after installation.
  --skip-environment-capture
                            Skip the machine-local diagnostic snapshot.
  --skip-lock-capture       Deprecated alias of --skip-environment-capture.
  --dry-run                 Print resolved paths and actions without changing files.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --behavior-repository) BEHAVIOR_REPOSITORY="$2"; shift 2 ;;
        --robomimic-repository) ROBOMIMIC_REPOSITORY="$2"; shift 2 ;;
        --git-proxy) GIT_PROXY="$2"; shift 2 ;;
        --prepare-only) PREPARE_ONLY=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        --skip-environment-capture|--skip-lock-capture) CAPTURE_ENVIRONMENT=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ENV_PREFIX="${CONDA_PREFIX:-}"
if [[ "$PREPARE_ONLY" != true && "$DRY_RUN" != true ]]; then
    command -v conda >/dev/null 2>&1 || { echo "Conda is required." >&2; exit 1; }
    if [[ -z "$ENV_PREFIX" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
        echo "Activate a dedicated Python 3.10 Conda environment before running setup." >&2
        echo "See README.md for environment creation instructions." >&2
        exit 2
    fi
    ENV_PREFIX="$(cd "$ENV_PREFIX" && pwd)"
    python -c 'import sys; assert sys.version_info[:2] == (3, 10), "ElogGen requires Python 3.10; got " + sys.version'
fi

THIRD_PARTY_ROOT="$DATA_ROOT/third_party"
BEHAVIOR_ROOT="$THIRD_PARTY_ROOT/BEHAVIOR-1K"
ROBOMIMIC_ROOT="$THIRD_PARTY_ROOT/robomimic"
CUROBO_ROOT="$THIRD_PARTY_ROOT/curobo"
DATASET_ROOT="$BEHAVIOR_ROOT/datasets"
CACHE_ROOT="$DATA_ROOT/cache"
PIP_CACHE_ROOT="$CACHE_ROOT/pip"
SOURCE_CACHE_ROOT="$CACHE_ROOT/sources"
ISAAC_WHEEL_CACHE="$CACHE_ROOT/isaacsim-4.5.0.0-wheels"
TMP_ROOT="$DATA_ROOT/tmp"
CUROBO_ARCHIVE="$SOURCE_CACHE_ROOT/curobo-$CUROBO_COMMIT.tar.gz"
EXPECTED_COMMIT="$(awk '/^commit:/ {print $2; exit}' "$PROJECT_ROOT/src/eloggen/patches/behavior1k/version.yaml")"
EXPECTED_ROBOMIMIC_COMMIT="$(awk '/^commit:/ {print $2; exit}' "$PROJECT_ROOT/src/eloggen/patches/robomimic/version.yaml")"

show_configuration() {
    printf '%s\n' \
        "ElogGen environment configuration" \
        "  project:          $PROJECT_ROOT" \
        "  data root:        $DATA_ROOT" \
        "  third-party root: $THIRD_PARTY_ROOT" \
        "  active env:       ${CONDA_PREFIX:-(activate a Python 3.10 Conda environment before setup)}" \
        "  BEHAVIOR source:  $BEHAVIOR_ROOT" \
        "  BEHAVIOR repo:    $BEHAVIOR_REPOSITORY" \
        "  BEHAVIOR commit:  $EXPECTED_COMMIT" \
        "  robomimic source: $ROBOMIMIC_ROOT" \
        "  robomimic repo:   $ROBOMIMIC_REPOSITORY" \
        "  robomimic commit: $EXPECTED_ROBOMIMIC_COMMIT" \
        "  Curobo source:    $CUROBO_ROOT" \
        "  Curobo commit:    $CUROBO_COMMIT" \
        "  dataset root:     $DATASET_ROOT" \
        "  Git proxy:        ${GIT_PROXY:-not configured}" \
        "  cache root:       $CACHE_ROOT" \
        "  Isaac wheel cache:$ISAAC_WHEEL_CACHE" \
        "  temporary files:  $TMP_ROOT" \
        "  CUDA/GCC:         toolkit 12.4, compiler 12.4 (inside env)" \
        "  runtime deps:     environments/requirements-runtime.txt" \
        "  env snapshot:     $([[ "$CAPTURE_ENVIRONMENT" == true ]] && echo machine-local || echo disabled)"
}

show_configuration
if [[ "$DRY_RUN" == true ]]; then
    printf '%s\n' \
        "[DRY-RUN] fetch or verify the pinned BEHAVIOR-1K source under third_party/" \
        "[DRY-RUN] fetch or verify the pinned robomimic source under third_party/" \
        "[DRY-RUN] apply the ElogGen OmniGibson/OpenArm/BDDL overlay" \
        "[DRY-RUN] verify the six included task HDF5 files" \
        "[DRY-RUN] download required BEHAVIOR and OmniGibson assets" \
        "[DRY-RUN] verify or download the checksum-pinned Curobo source archive" \
        "[DRY-RUN] install Curobo source as third_party/curobo" \
        "[DRY-RUN] install dependencies into the active Python 3.10 Conda environment" \
        "[DRY-RUN] install OmniGibson, BDDL and primitives" \
        "[DRY-RUN] install ElogGen and run validation" \
        "$([[ "$CAPTURE_ENVIRONMENT" == true ]] && echo '[DRY-RUN] capture a machine-local environment snapshot' || true)"
    exit 0
fi

mkdir -p "$THIRD_PARTY_ROOT" "$PIP_CACHE_ROOT" "$SOURCE_CACHE_ROOT" "$ISAAC_WHEEL_CACHE" "$TMP_ROOT"

GIT_FETCH_OPTIONS=(-c http.version=HTTP/1.1)
if [[ -n "$GIT_PROXY" ]]; then
    GIT_FETCH_OPTIONS+=(-c "http.proxy=$GIT_PROXY")
fi

fetch_pinned_source() {
    local label="$1"
    local repository="$2"
    local commit="$3"
    local destination="$4"
    local temp_prefix="$5"

    if [[ -d "$destination/.git" ]]; then
        return 0
    fi
    if [[ -e "$destination" ]]; then
        echo "$label destination exists but is not a git checkout: $destination" >&2
        return 1
    fi

    local work_root
    local checkout_root
    local fetch_ok=false
    work_root="$(mktemp -d "$THIRD_PARTY_ROOT/.${temp_prefix}-fetch.XXXXXX")"
    checkout_root="$work_root/source"

    cleanup_pinned_source() {
        if [[ -n "${work_root:-}" && -d "$work_root" ]]; then
            rm -rf -- "$work_root"
        fi
    }
    trap cleanup_pinned_source EXIT

    git init -q "$checkout_root"
    git -C "$checkout_root" remote add origin "$repository"

    for attempt in 1 2 3; do
        echo "Fetching pinned $label source (attempt $attempt/3)..."
        if git "${GIT_FETCH_OPTIONS[@]}" -C "$checkout_root" fetch \
            --depth=1 --no-tags origin "$commit"; then
            fetch_ok=true
            break
        fi
        sleep $((attempt * 2))
    done
    if [[ "$fetch_ok" != true ]]; then
        echo "Failed to fetch pinned $label source after three attempts." >&2
        return 1
    fi

    git -C "$checkout_root" checkout --detach FETCH_HEAD
    local actual_commit
    actual_commit="$(git -C "$checkout_root" rev-parse HEAD)"
    if [[ "$actual_commit" != "$commit" ]]; then
        echo "$label fetched commit mismatch: expected=$commit actual=$actual_commit" >&2
        return 1
    fi

    mv "$checkout_root" "$destination"
    rmdir "$work_root"
    work_root=""
    trap - EXIT
}

fetch_pinned_source \
    "BEHAVIOR-1K" \
    "$BEHAVIOR_REPOSITORY" \
    "$EXPECTED_COMMIT" \
    "$BEHAVIOR_ROOT" \
    "behavior"

ACTUAL_COMMIT="$(git -C "$BEHAVIOR_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "BEHAVIOR commit mismatch: expected=$EXPECTED_COMMIT actual=$ACTUAL_COMMIT" >&2
    exit 1
fi

fetch_pinned_source \
    "robomimic" \
    "$ROBOMIMIC_REPOSITORY" \
    "$EXPECTED_ROBOMIMIC_COMMIT" \
    "$ROBOMIMIC_ROOT" \
    "robomimic"

ACTUAL_ROBOMIMIC_COMMIT="$(git -C "$ROBOMIMIC_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_ROBOMIMIC_COMMIT" != "$EXPECTED_ROBOMIMIC_COMMIT" ]]; then
    echo "robomimic commit mismatch: expected=$EXPECTED_ROBOMIMIC_COMMIT actual=$ACTUAL_ROBOMIMIC_COMMIT" >&2
    exit 1
fi

python3 "$PROJECT_ROOT/scripts/apply_patches.py" \
    --behavior-root "$BEHAVIOR_ROOT" \
    --robomimic-root "$ROBOMIMIC_ROOT" \
    --apply

python3 "$PROJECT_ROOT/scripts/prepare_resources.py" \
    --group task-data \
    --check

if [[ "$PREPARE_ONLY" == true ]]; then
    echo "Source preparation complete; environment installation was skipped."
    exit 0
fi

if [[ ! -f "$CUROBO_ARCHIVE" ]]; then
    CUROBO_DOWNLOAD_TMP="$(mktemp "$SOURCE_CACHE_ROOT/.curobo-download.XXXXXX")"
    cleanup_curobo_download() {
        if [[ -n "${CUROBO_DOWNLOAD_TMP:-}" && -f "$CUROBO_DOWNLOAD_TMP" ]]; then
            rm -f -- "$CUROBO_DOWNLOAD_TMP"
        fi
    }
    trap cleanup_curobo_download EXIT
    curl --fail --location \
        --retry 10 --retry-all-errors --retry-delay 3 \
        --connect-timeout 30 \
        --output "$CUROBO_DOWNLOAD_TMP" \
        "$CUROBO_ARCHIVE_URL"
    echo "$CUROBO_ARCHIVE_SHA256  $CUROBO_DOWNLOAD_TMP" | sha256sum --check --status
    mv "$CUROBO_DOWNLOAD_TMP" "$CUROBO_ARCHIVE"
    CUROBO_DOWNLOAD_TMP=""
    trap - EXIT
fi
echo "$CUROBO_ARCHIVE_SHA256  $CUROBO_ARCHIVE" | sha256sum --check --status || {
    echo "Curobo source archive checksum mismatch: $CUROBO_ARCHIVE" >&2
    exit 1
}

if [[ ! -d "$CUROBO_ROOT" ]]; then
    CUROBO_EXTRACT_TMP="$(mktemp -d "$THIRD_PARTY_ROOT/.curobo-extract.XXXXXX")"
    cleanup_curobo_extract() {
        if [[ -n "${CUROBO_EXTRACT_TMP:-}" && -d "$CUROBO_EXTRACT_TMP" ]]; then
            rm -rf -- "$CUROBO_EXTRACT_TMP"
        fi
    }
    trap cleanup_curobo_extract EXIT
    tar -xzf "$CUROBO_ARCHIVE" --strip-components=1 -C "$CUROBO_EXTRACT_TMP"
    printf '%s\n' "$CUROBO_COMMIT" > "$CUROBO_EXTRACT_TMP/.eloggen-source-commit"
    mv "$CUROBO_EXTRACT_TMP" "$CUROBO_ROOT"
    CUROBO_EXTRACT_TMP=""
    trap - EXIT
fi
if [[ "$(cat "$CUROBO_ROOT/.eloggen-source-commit" 2>/dev/null || true)" != "$CUROBO_COMMIT" ]]; then
    echo "Curobo source marker mismatch: $CUROBO_ROOT" >&2
    echo "Remove that directory and rerun setup to install the pinned source revision." >&2
    exit 1
fi
echo "[OK] checksum-pinned Curobo source $CUROBO_COMMIT"

export PIP_CACHE_DIR="$PIP_CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export ELOGGEN_ISAAC_WHEEL_CACHE="$ISAAC_WHEEL_CACHE"
export TMPDIR="$TMP_ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
# GitHub occasionally terminates HTTP/2 TLS streams on this host. These
# process-local settings also reach git subprocesses launched by pip.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1
unset EXP_PATH CARB_APP_PATH ISAAC_PATH
# Never inherit a machine-specific package snapshot as an installation
# constraint. ElogGen's install definition is requirements-runtime.txt plus
# the explicitly pinned special dependencies below.
unset PIP_CONSTRAINT

# Avoid an unnecessary repodata request when the requested toolchain is
# already installed in the active environment.
INSTALLED_CUDA_VERSION="$("$ENV_PREFIX/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -1 || true)"
INSTALLED_GCC_VERSION="$("$ENV_PREFIX/bin/gcc" -dumpfullversion -dumpversion 2>/dev/null || true)"
if [[ "$INSTALLED_CUDA_VERSION" == 12.4* && "$INSTALLED_GCC_VERSION" == 12.4* ]]; then
    echo "Using CUDA $INSTALLED_CUDA_VERSION / GCC $INSTALLED_GCC_VERSION toolchain."
else
    set +u
    conda install --yes --prefix "$ENV_PREFIX" --channel conda-forge \
        "cuda-toolkit=12.4" "gcc=12.4" "gxx=12.4"
    set -u
fi
export CUDA_HOME="$ENV_PREFIX"
export CUDACXX="$ENV_PREFIX/bin/nvcc"
export CC="$ENV_PREFIX/bin/gcc"
export CXX="$ENV_PREFIX/bin/g++"
CUDA_TARGET_DIR="$ENV_PREFIX/targets/x86_64-linux"
export CPATH="$CUDA_TARGET_DIR/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_TARGET_DIR/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_TARGET_DIR/lib:$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_TARGET_DIR/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CUDACXX" --version | tail -1
"$CC" --version | head -1
python -m pip install numpy==1.26.4 ninja==1.13.0
python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url "$TORCH_INDEX_URL"

echo "Installing pinned Curobo from third_party/curobo..."
SETUPTOOLS_SCM_PRETEND_VERSION="$CUROBO_VERSION" \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="$CUROBO_VERSION" \
    python -m pip install --no-build-isolation "$CUROBO_ROOT"

behavior_install_ok=false
behavior_args=(
    --omnigibson
    --bddl
    --primitives
    --accept-nvidia-eula
)
for attempt in 1 2 3; do
    echo "Installing pinned BEHAVIOR runtime (attempt $attempt/3)..."
    if (
        cd "$BEHAVIOR_ROOT"
        bash ./setup.sh "${behavior_args[@]}"
    ); then
        behavior_install_ok=true
        break
    fi
    sleep $((attempt * 3))
done
if [[ "$behavior_install_ok" != true ]]; then
    echo "Failed to install the pinned BEHAVIOR runtime after three attempts." >&2
    exit 1
fi

python3 "$PROJECT_ROOT/scripts/prepare_resources.py" \
    --group task-data \
    --check

python -m pip install -r "$PROJECT_ROOT/environments/requirements-runtime.txt"
python -m pip install -e "$ROBOMIMIC_ROOT" --no-deps
python -m pip install -e "$PROJECT_ROOT" --no-deps

# ElogGen uses its packaged task scenes and does not need the upstream 2025
# Challenge task-instance bundle. Download only the shared runtime assets.
python -c 'from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()'
python -c 'from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)'

export ELOGGEN_ROOT="$PROJECT_ROOT"
export ELOGGEN_DATA_ROOT="$DATA_ROOT"
export ELOGGEN_THIRD_PARTY_ROOT="$THIRD_PARTY_ROOT"
export ELOGGEN_ENV_PREFIX="$ENV_PREFIX"
export ELOGGEN_BEHAVIOR_ROOT="$BEHAVIOR_ROOT"
export ELOGGEN_ROBOMIMIC_ROOT="$ROBOMIMIC_ROOT"
export ELOGGEN_DATASET_ROOT="$DATASET_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$BEHAVIOR_ROOT/OmniGibson:$ROBOMIMIC_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python "$PROJECT_ROOT/scripts/check_sources.py"
python -m pip check
python -c 'import eloggen; import omnigibson; import bddl; print("Runtime imports: OK")'
if [[ "$SKIP_TESTS" != true ]]; then
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
        -s "$PROJECT_ROOT/tests/unit" \
        -v
fi

if [[ "$CAPTURE_ENVIRONMENT" == true ]]; then
    python "$PROJECT_ROOT/scripts/capture_environment.py" \
        --behavior-root "$BEHAVIOR_ROOT" \
        --robomimic-root "$ROBOMIMIC_ROOT" \
        --output-dir "$PROJECT_ROOT/.eloggen/environment"
fi

echo "Environment installation complete."
echo "In a new shell, activate the same Conda environment first."
echo "Then load ElogGen runtime paths with: source $PROJECT_ROOT/scripts/activate_env.sh"