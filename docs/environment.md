# Environment setup

ElogGen installs into a Python 3.10 Conda environment created and activated by the user. The project scripts never create a Conda environment and do not decide where Conda stores environments.

## Prerequisites

- Linux on x86-64
- NVIDIA GPU and a driver compatible with Isaac Sim 4.5
- Git
- Conda
- standard compiler and archive tools

## Standard installation

Create and activate the environment through the machine's existing Conda installation:

```bash
conda env create -f environments/eloggen-base.yaml
conda activate eloggen

git clone https://github.com/Nano-Ping/ElogGen.git
cd ElogGen

bash setup.sh
```

The setup script requires a non-base Python 3.10 Conda environment to be active. It installs CUDA Toolkit 12.4, GCC/G++ 12.4, PyTorch, Curobo, the pinned BEHAVIOR runtime, ElogGen, and all runtime dependencies into that environment.

In each new shell:

```bash
conda activate eloggen
cd /path/to/ElogGen
source scripts/activate_env.sh
```

`activate_env.sh` does not activate Conda. It only validates the current environment and loads ElogGen's runtime paths.

## Resource locations

Without an override, ElogGen groups all third-party source checkouts under `third_party/` and keeps caches and temporary setup files separate:

```text
ElogGen/
├── third_party/
│   ├── BEHAVIOR-1K/
│   │   └── datasets/
│   ├── robomimic/
│   └── curobo/
├── cache/
└── tmp/
```

The Curobo source directory intentionally uses the stable name `curobo`. Its exact source revision is still pinned by `CUROBO_COMMIT`, verified through the downloaded archive SHA256, and recorded in `third_party/curobo/.eloggen-source-commit`.

This location is independent of the Conda environment. `cache/` stores reusable package, source, and download files; `tmp/` holds temporary setup files. Both are created automatically and can be removed when no setup is running. Users who prefer another disk can select it explicitly:

```bash
bash setup.sh \
  --data-root /path/to/storage/ElogGen
```

The same structure is then created below `/path/to/storage/ElogGen`, including `/path/to/storage/ElogGen/third_party`. The setup command records the selected paths in the ignored `.eloggen.local` file.

## Source access

The default source checkouts use GitHub HTTPS. Use SSH or mirror URLs when needed:

```bash
bash setup.sh \
  --behavior-repository git@github.com:StanfordVL/BEHAVIOR-1K.git \
  --robomimic-repository git@github.com:ChengshuLi/robomimic.git
```

For Git clones through an HTTP(S) proxy, add `--git-proxy http://host:port`. Package and Hugging Face downloads use the standard `HTTP_PROXY` and `HTTPS_PROXY` environment variables.

## What setup performs

1. Verify the six task HDF5 files included in the repository.
2. Clone the pinned BEHAVIOR-1K source revision into `third_party/BEHAVIOR-1K`.
3. Clone the pinned robomimic source revision into `third_party/robomimic`.
4. Prepare the pinned Curobo source under `third_party/curobo`.
5. Apply the versioned ElogGen BEHAVIOR/OpenArm and robomimic adaptations.
6. Download the required BEHAVIOR and OmniGibson assets. The unrelated 2025 Challenge task-instance bundle is not installed.
7. Install the pinned CUDA, compiler, PyTorch, Curobo, BEHAVIOR, robomimic, and ElogGen dependencies into the active environment.
8. Run dependency, import, task-data, and unit checks.

OpenArm runtime assets are deliberately separate from environment setup. Install them only after the environment is ready:

```bash
conda activate eloggen
cd /path/to/ElogGen
source scripts/activate_env.sh

eloggen resources install openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT"
```

The manifest downloads the files from the public, pinned
[Nano-Ping/ElogGen-Assets](https://huggingface.co/datasets/Nano-Ping/ElogGen-Assets)
dataset and verifies their sizes and SHA256 hashes. Set
`ELOGGEN_OPENARM_ASSET_BASE_URL` only when using a mirror with the same layout.

The pinned robomimic checkout is installed in editable mode from `third_party/robomimic`. The upstream package is not copied into `src/eloggen`.

## Existing installations

Older ElogGen installations may still have:

```text
ElogGen/
├── BEHAVIOR-1K/
├── robomimic/
└── deps/curobo-<commit>/
```

An existing working Conda environment does not need to be rebuilt. Existing `.eloggen.local` paths remain usable until those source directories are migrated. Fresh installations use `third_party/` by default.

## Validation

After installation:

```bash
conda activate eloggen
cd /path/to/ElogGen
source scripts/activate_env.sh
bash scripts/validate_environment.sh
```

Add `--with-simulator` to verify the separately installed OpenArm assets and include a visible one-step OmniGibson startup test. Full dataset generation is not part of this validation.

## Reproducibility records

`scripts/capture_environment.py` records package and source revisions under `.eloggen/environment/`. These files are diagnostic evidence only, are excluded from Git, and are not used as installation constraints.
