# Environment layout

ElogGen keeps installation inputs, third-party source checkouts, caches, and machine-local diagnostics separate.

## Installation sources of truth

- `eloggen-base.yaml`: minimal Conda bootstrap environment (Python 3.10 and packaging tools).
- `requirements-runtime.txt`: pinned direct Python runtime dependencies used by ElogGen.
- `scripts/setup_env.sh`: installs special dependencies that need explicit versions, indexes, source commits, patches, licenses, or compiler toolchains. This includes CUDA/GCC, PyTorch, Curobo, BEHAVIOR-1K / OmniGibson, BDDL, and robomimic.
- `src/eloggen/patches/*/version.yaml`: pins upstream BEHAVIOR-1K and robomimic commits used by the runtime.

Machine-generated `pip freeze` and `conda list --explicit` outputs are intentionally **not** installation constraints. They can contain host-specific packages such as ROS packages and therefore should not define a clean ElogGen environment.

## Third-party source layout

All external source trees are grouped under one ignored directory:

```text
third_party/
├── BEHAVIOR-1K/
├── robomimic/
└── curobo/
```

The Curobo directory deliberately uses the stable name `curobo` rather than embedding a commit hash in the directory name. Reproducibility is enforced by the pinned commit and the `.eloggen-source-commit` marker written by setup.

When `--data-root /path/to/storage/ELogGen` is used, the same layout is created under `/path/to/storage/ELogGen/third_party`.

## Machine-local environment snapshot

At the end of setup, `scripts/capture_environment.py` records diagnostic evidence under:

```text
.eloggen/environment/
├── environment-manifest.json
├── pip-freeze.snapshot.txt
└── conda-explicit.snapshot.txt
```

This directory is ignored by Git. These files are useful for debugging or recording a particular machine, but they are not used by `setup_env.sh` to constrain future installations.

Set `ELOGGEN_ENV_SNAPSHOT_DIR` or pass `--output-dir` to `capture_environment.py` to place snapshots elsewhere. Use `--skip-environment-capture` with `setup.sh` / `scripts/setup_env.sh` to skip snapshot creation entirely. The old `--skip-lock-capture` option remains as a compatibility alias.

## Existing environments

Changing the external-source layout does not require rebuilding an existing working `eloggen` Conda environment. A machine that already has the old root-level `BEHAVIOR-1K/`, `robomimic/`, or `deps/curobo-*` layout can continue to use its existing `.eloggen.local` until those source directories are migrated.

For a fresh environment, create the bootstrap environment and then run setup:

```bash
conda env create -f environments/eloggen-base.yaml
conda activate eloggen
bash setup.sh
```

Fresh setup writes the canonical `third_party` locations to `.eloggen.local`.
