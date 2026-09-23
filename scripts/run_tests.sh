#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Some workstation images expose unrelated system pytest plugins (for example
# ROS launch_testing) through PYTHONPATH.  ElogGen tests are intentionally run
# with only the plugins installed in the selected project environment.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest "$@"
