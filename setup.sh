#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$PROJECT_ROOT/.eloggen.local" ]]; then
    source "$PROJECT_ROOT/.eloggen.local"
fi
DATA_ROOT="${ELOGGEN_DATA_ROOT:-$PROJECT_ROOT}"
DRY_RUN=false

arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    case "${arguments[$index]}" in
        --data-root) DATA_ROOT="${arguments[$((index + 1))]}" ;;
        --dry-run|-h|--help) DRY_RUN=true ;;
    esac
done

# Fast path for pinned GitHub sources: download the commit archive through
# codeload.github.com, then attach only lightweight Git metadata. This keeps
# the exact pinned commit semantics needed by the patch / validation tools
# without transferring all source blobs through Git Smart HTTP.
bash "$PROJECT_ROOT/scripts/prepare_pinned_sources.sh" "$@"

bash "$PROJECT_ROOT/scripts/setup_env.sh" "$@"

if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

THIRD_PARTY_ROOT="$DATA_ROOT/third_party"
BEHAVIOR_ROOT="$THIRD_PARTY_ROOT/BEHAVIOR-1K"
ROBOMIMIC_ROOT="$THIRD_PARTY_ROOT/robomimic"
DATASET_ROOT="$BEHAVIOR_ROOT/datasets"
{
    printf 'export ELOGGEN_DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'export ELOGGEN_THIRD_PARTY_ROOT=%q\n' "$THIRD_PARTY_ROOT"
    printf 'export ELOGGEN_BEHAVIOR_ROOT=%q\n' "$BEHAVIOR_ROOT"
    printf 'export ELOGGEN_ROBOMIMIC_ROOT=%q\n' "$ROBOMIMIC_ROOT"
    printf 'export ELOGGEN_DATASET_ROOT=%q\n' "$DATASET_ROOT"
    if [[ -n "${ELOGGEN_OPENARM_ASSET_BASE_URL:-}" ]]; then
        printf 'export ELOGGEN_OPENARM_ASSET_BASE_URL=%q\n' "$ELOGGEN_OPENARM_ASSET_BASE_URL"
    fi
} > "$PROJECT_ROOT/.eloggen.local"

echo "Saved machine-local paths to $PROJECT_ROOT/.eloggen.local"
echo "In a new shell, activate your Conda environment first."
echo "Then load runtime paths with: source $PROJECT_ROOT/scripts/activate_env.sh"
