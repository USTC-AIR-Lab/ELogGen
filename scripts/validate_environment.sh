#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPETITIONS=1
WITH_SIMULATOR=false
TASK="openarm_real_exp_1"

usage() {
    cat <<'EOF'
Usage: bash scripts/validate_environment.sh [options]

Options:
  --repetitions N     Run the complete validation N consecutive times.
  --with-simulator    Include a visible one-step OmniGibson startup smoke test.
  --task NAME         Task used for simulator startup (default: openarm_real_exp_1).
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --with-simulator) WITH_SIMULATOR=true; shift ;;
        --task) TASK="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repetitions must be a positive integer" >&2
    exit 2
fi

source "$PROJECT_ROOT/scripts/activate_env.sh"
cd "$PROJECT_ROOT"

for ((pass = 1; pass <= REPETITIONS; pass++)); do
    echo "===== ElogGen validation pass $pass/$REPETITIONS ====="
    behavior_check_args=()
    if [[ "$WITH_SIMULATOR" == true ]]; then
        python scripts/prepare_resources.py \
            --group openarm-assets \
            --dataset-root "$ELOGGEN_DATASET_ROOT" \
            --check
        behavior_check_args+=(--with-openarm-assets)
    fi
    python scripts/check_sources.py "${behavior_check_args[@]}"
    python -m pip check
    python - <<'PY'
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

env_prefix = Path(os.environ["ELOGGEN_ENV_PREFIX"]).resolve()
behavior_root = Path(os.environ["ELOGGEN_BEHAVIOR_ROOT"]).resolve()
robomimic_root = Path(os.environ["ELOGGEN_ROBOMIMIC_ROOT"]).resolve()
assert Path(sys.prefix).resolve() == env_prefix, (sys.prefix, env_prefix)

expected = {
    "eloggen": Path(os.environ["ELOGGEN_ROOT"]).resolve(),
    "omnigibson": behavior_root / "OmniGibson",
    "bddl": behavior_root / "bddl",
    "robomimic": robomimic_root,
}
for module, root in expected.items():
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, module
    origin = Path(spec.origin).resolve()
    assert origin.is_relative_to(root), f"{module} resolved outside {root}: {origin}"
    print(f"[OK] {module}: {origin}")
PY
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/unit -v

    for task_dir in "$PROJECT_ROOT/src/eloggen/datasets/taskpacks"/*; do
        [[ -f "$task_dir/task.yaml" ]] || continue
        python -m eloggen.cli check "$(basename "$task_dir")"
    done

    run_dir="/tmp/eloggen-phase2-validation-$pass"
    python -m eloggen.cli pipeline \
        --config "$PROJECT_ROOT/src/eloggen/datasets/examples/openarm_real_exp_1.json" \
        --stages task_graph,logic,recipe,export_config,generate \
        --run-dir "$run_dir" \
        --dry-run

    if [[ "$WITH_SIMULATOR" == true ]]; then
        unset OMNIGIBSON_HEADLESS
        export OMNIGIBSON_NO_OMNI_LOGS=True
        python -m eloggen.cli scene "$TASK" --steps 1
    fi
    echo "===== ElogGen validation pass $pass/$REPETITIONS: PASS ====="
done

echo "ElogGen completed $REPETITIONS consecutive validation pass(es)."
