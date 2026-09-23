# Asset provenance and release boundary

## External assets

BEHAVIOR-1K object meshes, scene assets, OpenArm robot assets, and
`omnigibson.key` remain in the external dataset directory and are not included
in the source archive. BEHAVIOR assets are installed during environment setup;
OpenArm robot assets are installed separately using a user-supplied source. Their
upstream licenses and access terms apply.

## Included integration files

- `src/eloggen/patches/behavior1k/openarm_bimanual.py` installs the OpenArm robot class
  into the pinned OmniGibson checkout.
- `src/eloggen/patches/behavior1k/patches` contains textual adaptations to the pinned
  BEHAVIOR-1K source.
- Task `scene.json`, BDDL, and camera files describe arrangements and references;
  they do not embed the external BH1K mesh files.

## Required owner confirmations

Before release, record the author/rightsholder and chosen terms for the OpenArm
robot integration and generated USDA assets. If any file was copied from a
third-party OpenArm repository, preserve that repository's license and attribution
instead of applying the ElogGen-original code license.
