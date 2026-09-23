# ElogGen architecture

## Design goal

ElogGen separates the reusable dataset-extension method from simulator compatibility code, task data, and private hardware collection.

## Dependency direction

```text
datasets / CLI
    -> pipeline
        -> planning
        -> generation
            -> generation_runtime
                -> external OmniGibson and robomimic
    -> simulation
```

`planning` contains simulator-independent task graphs, constraints, skills, and recipes. `pipeline` parses source data and orchestrates stages. `generation` converts pipeline output into runtime calls. `generation_runtime` implements trajectory composition, scheduling, context injection, dataset writing, and OmniGibson task interfaces. `simulation` owns startup, scene viewing, camera tuning, and asset inspection.

External source adaptations are isolated under `patches`. Setup checks out pinned BEHAVIOR-1K, robomimic, and Curobo sources under `third_party/` and applies the versioned adaptations idempotently.

No runtime module may rely on a developer-specific source checkout or absolute
machine path.

## Task packs

Each directory under `src/eloggen/datasets/taskpacks/` is a release unit.
`task.yaml` references:

- generation configuration
- scene template
- BDDL problem
- camera mounts
- source and processed HDF5
- source hashes and collection-interface metadata

This makes provenance and validation local to the task instead of scattering paths across scripts.

## BEHAVIOR-1K boundary

BEHAVIOR-1K and robomimic remain external upstream dependencies. Their `version.yaml` files pin the expected revisions. `scripts/apply_patches.py` verifies each checkout, applies missing patches, installs the OpenArm robot definition and task BDDL files, and overlays the adapted robomimic environment.

## Collection boundary

`eloggen.teleoperation` is the public hardware interface. Real teleoperation belongs in a separately distributed plugin and may depend on ROS2, Pico, and private device packages. Scene visualization and camera tuning remain public because they are simulator workflows and contain no device collection implementation.
