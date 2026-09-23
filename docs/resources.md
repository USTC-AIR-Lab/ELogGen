# External resources

The Git repository contains all Python code, task configuration, BDDL, scenes,
camera settings, the six example HDF5 files, small generated USDA assets,
environment scripts, and tests. Large or separately distributed runtime data is
installed after environment setup through `eloggen resources`.

## Repository boundary

The following files are intentionally not stored in Git:

- the 311 MB OpenArm robot USD and its four metadata/Curobo configuration files
- standard BEHAVIOR-1K datasets and assets

Small task-local USDA assets such as the handled tray and angled brush dock stay
inside `src/eloggen` and require no separate download.

Every example HDF5 and ElogGen-specific external file has an exact destination,
byte size, and SHA256 in `src/eloggen/datasets/manifest.json`. Task manifests use
package-relative HDF5 paths, so generation configuration does not depend on a
machine's absolute directory layout.

## Download configuration

Environment installation never downloads OpenArm resources. The manifest contains
the public, pinned Hugging Face URLs. Install and verify them after activating
ElogGen:

```bash
source scripts/activate_env.sh
eloggen resources install openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT"
```

The following environment variables remain available as advanced mirrors and
offline-machine overrides:

- `ELOGGEN_DATA_ROOT`: optional location for BEHAVIOR source and download caches
- `ELOGGEN_DATASET_ROOT`: BEHAVIOR dataset directory created by setup
- ELOGGEN_OPENARM_ASSET_BASE_URL: optional mirror directory URL containing the OpenArm layout; it overrides the manifest URLs

For an optional mirror, the tool appends the manifest path, for example:

```text
${ELOGGEN_OPENARM_ASSET_BASE_URL}/custom_dataset/objects/robot/openarmbimanual/usd/openarmbimanual.usda
```

The default installation needs no URL argument. A mirror can be selected with:

```bash
eloggen resources install openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT" \
  --base-url "$ELOGGEN_OPENARM_ASSET_BASE_URL"

eloggen resources check openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT"
```

An unpacked OpenArm bundle can be used instead of URLs. Its top level contains
`dataset/custom_dataset/...`:

```bash
eloggen resources install openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT" \
  --source-root /path/to/unpacked/eloggen-resources
```

Downloads and copies use temporary files, verify size and SHA256, and only then
move into the final location. Existing files with a wrong hash are never silently
overwritten.

## BEHAVIOR data

ElogGen downloads the required BEHAVIOR and OmniGibson assets into the configured
data root. The 2025 Challenge task-instance bundle is not used by the three
packaged ElogGen tasks and is not downloaded.

Environment setup and OpenArm resource installation are independent. The default
OpenArm URLs are already present in the manifest:

```bash
conda activate eloggen
bash setup.sh

source scripts/activate_env.sh
eloggen resources install openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT"
```
