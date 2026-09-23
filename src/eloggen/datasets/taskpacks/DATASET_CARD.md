# ElogGen source demonstration dataset card

## Scope

ElogGen provides one raw and one processed OmniGibson/robomimic-style HDF5
demonstration per task directly in the repository. These files provide simulator
state and action trajectories used to synthesize additional demonstrations.
They do not bundle the BEHAVIOR-1K asset dataset.

| Task | Raw frames | Processed frames | Action width |
| --- | ---: | ---: | ---: |
| `openarm_drawer_storage` | 2307 | 1901 | 16 |
| `openarm_fruit_basket_bagging` | 2127 | 1876 | 16 |
| `openarm_real_exp_1` | 1480 | 1480 | 16 |

The Git repository contains these six HDF5 payloads at stable relative paths.
`src/eloggen/datasets/manifest.json` and each task's `task.yaml` record SHA256
checksums. Processed files contain `datagen_info`; the drawer task additionally
contains phase annotations. HDF5 attributes use task-relative paths rather than
developer-machine paths.

## Collection

The demonstrations were collected with the OpenArm task interface. Public task
packs expose the `eloggen.collectors.v1` contract, but the real Pico/ROS/OpenArm
teleoperation implementation is not included.

## Intended use

The files are intended as source demonstrations for ElogGen's task-graph,
recipe, replay, and dataset-generation pipeline. They are not a benchmark split
and should not be interpreted as independently sampled training episodes.

## Dependencies

Replaying or generating from these demonstrations requires the pinned
BEHAVIOR-1K/OmniGibson checkout and external assets prepared by
`scripts/setup_env.sh`. Dataset and asset licenses remain separate.

## Installation

No separate source-HDF5 download is required. To verify the bundled files, run
`python3 scripts/prepare_resources.py --group task-data --check --project-root .`.
See `docs/resources.md` for separately distributed simulator assets.
