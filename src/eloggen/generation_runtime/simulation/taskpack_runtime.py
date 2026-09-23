"""Build OmniGibson runtime interfaces from task-pack manifests.

This module is the compatibility bridge between task-owned configuration and the
legacy class registry used by the inherited generation runtime. New ordinary
OpenArm tasks should not need a Python interface class: declaring tracked
objects (and optional articulated targets) in task.yaml is sufficient.
"""

from __future__ import annotations

from typing import Any

from eloggen.pipeline.task_pack import load_task_pack
from eloggen.generation_runtime.simulation.drawers import DrawerTarget
from eloggen.generation_runtime.simulation.omnigibson import (
    TASK_CONFIGS,
    OmniGibsonInterfaceBimanual,
    TaskConfig,
)


GENERIC_TASKPACK_INTERFACE = "EG_TaskPackOmniGibsonInterface"


def _tracked_objects(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(registry_name) for key, registry_name in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(name): str(name) for name in value}
    raise TypeError("runtime.interface.tracked_objects must be a mapping or list")


def _articulated_targets(value: Any) -> dict[str, DrawerTarget]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("runtime.interface.articulated_targets must be a mapping")

    targets: dict[str, DrawerTarget] = {}
    for target_id, raw in value.items():
        config = dict(raw or {})
        config["handle_local_position"] = tuple(config.get("handle_local_position") or (0.0, 0.0, 0.0))
        config["handle_local_orientation_xyzw"] = tuple(
            config.get("handle_local_orientation_xyzw") or (0.0, 0.0, 0.0, 1.0)
        )
        targets[str(target_id)] = DrawerTarget(target_id=str(target_id), **config)
    return targets


def task_config_from_task_pack(task_name: str) -> TaskConfig:
    task = load_task_pack(task_name)
    runtime_interface = dict(task.runtime.get("interface") or {})
    return TaskConfig(
        name=task.name,
        tracked_objects=_tracked_objects(runtime_interface.get("tracked_objects")),
        termination_signals=dict(runtime_interface.get("termination_signals") or {}),
        robot_specific_objects=dict(runtime_interface.get("robot_specific_objects") or {}),
        bimanual=bool(dict(task.runtime.get("robot") or {}).get("bimanual", True)),
        articulated_targets=_articulated_targets(runtime_interface.get("articulated_targets")),
    )


def _append_prefix(module: Any, attribute: str, task_name: str) -> None:
    """Extend an inherited tuple registry without requiring another source edit."""
    current = tuple(getattr(module, attribute, ()))
    if task_name not in current:
        setattr(module, attribute, (*current, task_name))


def _install_legacy_runtime_compatibility(task) -> None:
    """Teach inherited modules about a task discovered from task.yaml.

    The compatibility profile is task-owned. ``runtime.robot.type: OpenArm`` opts
    the task into fixed-base OpenArm dispatch; ``openarm_cashier_style`` opts it
    into the source-state alignment used by the bundled checkout-style tasks.
    This keeps those legacy branches available without asking task authors to
    edit a Python registry.
    """
    task_name = task.name
    robot = dict(task.runtime.get("robot") or {})
    compatibility = dict(task.runtime.get("compatibility") or {})
    is_openarm = str(robot.get("type") or "").lower() == "openarm"

    if is_openarm:
        try:
            from eloggen.generation_runtime.generator import generator as generator_module
            _append_prefix(generator_module, "OPENARM_TASK_PREFIXES", task_name)
        except Exception:
            # The generator may not be imported yet during source preparation.
            pass

    try:
        import robomimic.envs.env_omnigibson as env_omnigibson_module
        if is_openarm:
            _append_prefix(env_omnigibson_module, "OPENARM_TASK_PREFIXES", task_name)
        if compatibility.get("openarm_cashier_style"):
            _append_prefix(env_omnigibson_module, "OPENARM_CASHIER_STYLE_PREFIXES", task_name)
    except Exception:
        # Robomimic is optional for lightweight configuration/unit-test paths.
        pass


def ensure_task_runtime_registered(task_name: str) -> TaskConfig:
    """Load task runtime metadata and expose it to legacy runtime consumers."""
    task = load_task_pack(task_name)
    task_config = task_config_from_task_pack(task.name)
    # TASK_CONFIGS is still imported by the robomimic compatibility wrapper.
    # Mutating the shared mapping here lets old code consume new task packs
    # without adding another hard-coded task entry.
    TASK_CONFIGS[task_config.name] = task_config
    _install_legacy_runtime_compatibility(task)
    return task_config


class EG_TaskPackOmniGibsonInterface(OmniGibsonInterfaceBimanual):
    """Generic bimanual interface whose object schema comes from task.yaml."""

    def __init__(self, env):
        runtime_name = getattr(env, "name", None) or getattr(getattr(env, "task", None), "activity_name", None)
        if not runtime_name:
            raise ValueError("Cannot infer task-pack name from OmniGibson environment")
        task_config = ensure_task_runtime_registered(str(runtime_name))
        super().__init__(env, task_config)
