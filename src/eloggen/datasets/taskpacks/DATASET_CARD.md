# ElogGen source demonstration dataset card

## Scope

ElogGen provides one raw and one processed OmniGibson/robomimic-style HDF5
demonstration per task as separately downloaded resources. These files provide simulator state and action
trajectories used to synthesize additional demonstrations. They do not bundle the
BEHAVIOR-1K asset dataset.

| Task | Raw frames | Processed frames | Action width |
| --- | ---: | ---: | ---: |
| `openarm_drawer_storage` | 2307 | 1901 | 16 |
| `openarm_fruit_basket_bagging` | 2127 | 1876 | 16 |
| `openarm_real_exp_1` | 1480 | 1480 | 16 |

The Git repository keeps their stable relative paths but does not contain the
HDF5 payloads. `src/eloggen/datasets/manifest.json` and each task's `task.yaml` record
SHA256 checksums. Processed files
contain `datagen_info`; the drawer task additionally contains phase annotations.
Legacy host paths in HDF5 attributes have been replaced with task-relative paths.
The untouched pre-sanitization files are retained only in the ignored local
backup directory `.baseline/hdf5-before-release-sanitize-20260913`.

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

Configure the published task-data base URL in `.eloggen.local`, then run
`python3 scripts/prepare_resources.py`. See `docs/resources.md` for online and
offline installation layouts.
