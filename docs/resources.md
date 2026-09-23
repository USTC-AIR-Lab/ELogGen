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

## OpenArm resource installation

Environment setup does not download OpenArm resources. Obtain the OpenArm
runtime assets separately from an open-source community, then unpack them into
a local directory with this layout:

```text
/path/to/unpacked/openarm-resources/
└── dataset/
    └── custom_dataset/
        └── objects/
            └── robot/
                └── openarmbimanual/
                    ├── usd/openarmbimanual.usda
                    ├── misc/metadata.json
                    └── curobo/
                        ├── openarmbimanual_description_curobo_default.yaml
                        ├── openarmbimanual_description_curobo_arm.yaml
                        └── openarmbimanual_description_curobo_arm_no_torso.yaml
```

After activating the environment, copy and verify these files using the
resource installer:

```bash
source scripts/activate_env.sh
eloggen resources install openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT" \
  --source-root /path/to/unpacked/openarm-resources
eloggen resources check openarm \
  --dataset-root "$ELOGGEN_DATASET_ROOT"
```

The destination is
`$ELOGGEN_DATASET_ROOT/custom_dataset/objects/robot/openarmbimanual/`.
Alternatively, place the five files there yourself and run the check command.
The exact file sizes and SHA256 hashes are recorded in
`src/eloggen/datasets/manifest.json`; files with mismatched hashes are
rejected. No OpenArm download URL is embedded in the repository.

Users with their own layout-compatible mirror may optionally supply
`--base-url` or `ELOGGEN_OPENARM_ASSET_BASE_URL`. Without an explicit local
source or mirror, the resource installer does not attempt a download.

## BEHAVIOR data

ElogGen downloads the required BEHAVIOR and OmniGibson assets into the configured
data root. The 2025 Challenge task-instance bundle is not used by the three
packaged ElogGen tasks and is not downloaded.

Environment setup and OpenArm resource installation are independent.
