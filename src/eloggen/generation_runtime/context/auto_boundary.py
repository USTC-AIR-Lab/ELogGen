"""V3 helpers for comparing or replacing manual ElogGen generation runtime boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np

from eloggen.generation_runtime.context.boundary import (
    DEFAULT_DETECTOR_CONFIG,
    DEFAULT_OBJECT,
    detect_action_boundaries,
    load_demo_signals,
)


VALID_AUTO_BOUNDARY_MODES = ("off", "compare_only", "replace_manual")


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    def sort_key(name: str):
        if name.startswith("demo_"):
            try:
                return (int(name.split("_")[-1]), name)
            except ValueError:
                pass
        return (10**9, name)

    return sorted(data_group.keys(), key=sort_key)


def _get_item(container: Any, key: str, default: Any = None) -> Any:
    try:
        return container[key]
    except Exception:
        return getattr(container, key, default)


def _set_item(container: Any, key: str, value: Any) -> None:
    try:
        container[key] = value
    except Exception:
        setattr(container, key, value)


def _iter_subtask_specs(task_spec: Any):
    for phase_name in sorted(list(task_spec.keys())):
        phase_spec = task_spec[phase_name]
        for arm_key in ("arm_left", "arm_right"):
            arm_spec = _get_item(phase_spec, arm_key, {})
            for subtask_name in sorted(list(arm_spec.keys())):
                yield phase_name, arm_key, subtask_name, arm_spec[subtask_name]


def _last_subtask_name(arm_spec: Any) -> Optional[str]:
    names = sorted(list(arm_spec.keys()))
    return names[-1] if names else None


def _sync_bimanual_phase_end_steps(task_spec: Any, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updates_by_key = {
        (item.get("phase"), item.get("arm_key"), item.get("subtask")): item
        for item in updates
    }
    sync_records = []

    for phase_name in sorted(list(task_spec.keys())):
        phase_spec = task_spec[phase_name]
        last_specs = []
        term_candidates = []
        sync_record = {"phase": phase_name, "arms": []}

        for arm_key in ("arm_left", "arm_right"):
            arm_spec = _get_item(phase_spec, arm_key, {})
            subtask_name = _last_subtask_name(arm_spec)
            if subtask_name is None:
                continue

            subtask_spec = arm_spec[subtask_name]
            update = updates_by_key.get((phase_name, arm_key, subtask_name))
            current_term = _get_item(subtask_spec, "subtask_term_step", None)
            if current_term is not None:
                term_candidates.append(int(current_term))

            last_specs.append((arm_key, subtask_name, subtask_spec, update))
            sync_record["arms"].append(
                {
                    "arm_key": arm_key,
                    "subtask": subtask_name,
                    "status": None if update is None else update.get("status"),
                    "current_subtask_term_step": current_term,
                }
            )

        if not term_candidates:
            continue

        canonical_term = max(term_candidates)
        sync_record["synced_subtask_term_step"] = canonical_term

        for arm_key, subtask_name, subtask_spec, update in last_specs:
            _set_item(subtask_spec, "subtask_term_step", canonical_term)

            if _get_item(subtask_spec, "object_ref", None) is None:
                mp_end = _get_item(subtask_spec, "MP_end_step", None)
                if mp_end is None or int(mp_end) != canonical_term:
                    _set_item(subtask_spec, "MP_end_step", canonical_term)
                    if update is not None:
                        update["synced_MP_end_step"] = canonical_term

            if update is not None:
                update["synced_subtask_term_step"] = canonical_term

        sync_records.append(sync_record)

    return sync_records


def _detect_for_subtask(
    source_dataset_path: str,
    demo_keys: Iterable[str],
    task_name: str,
    subtask_spec: Any,
    max_manual_auto_diff: int,
) -> Dict[str, Any]:
    arm = _get_item(subtask_spec, "arm", "left")
    action = _get_item(subtask_spec, "action", None) or "pick"
    object_name = _get_item(subtask_spec, "object_ref", None)
    if object_name is None:
        return {
            "status": "skipped",
            "reason": "object_ref is null",
            "action": action,
            "arm": arm,
            "object_ref": None,
        }

    manual_mp = _get_item(subtask_spec, "MP_end_step", None)
    manual_term = _get_item(subtask_spec, "subtask_term_step", None)
    detector_config = {
        "manual_subtask": {
            "MP_end_step": manual_mp,
            "subtask_term_step": manual_term,
        }
    }
    for key in ("object_label_zh", "object_label_en", "target_label_zh", "target_label_en"):
        value = _get_item(subtask_spec, key, None)
        if value is not None:
            detector_config[key] = value
    target_ref = _get_item(subtask_spec, "target_ref", None)
    if target_ref is not None:
        detector_config["target_object"] = target_ref

    per_demo = []
    for demo_key in demo_keys:
        signals = load_demo_signals(
            hdf5_path=source_dataset_path,
            demo_key=demo_key,
            arm=arm,
            object_name=object_name or DEFAULT_OBJECT,
        )
        boundaries = detect_action_boundaries(signals, action=action, config=detector_config)
        boundaries.task = task_name
        detected_mp = boundaries.events.get("approach_end")
        detected_term = boundaries.events.get("subtask_end")
        per_demo.append(
            {
                "demo_key": demo_key,
                "detector": boundaries.detector,
                "detected_MP_end_step": detected_mp,
                "detected_subtask_term_step": detected_term,
                "events": boundaries.events,
                "metrics": boundaries.metrics,
            }
        )

    mp_values = [item["detected_MP_end_step"] for item in per_demo if item["detected_MP_end_step"] is not None]
    term_values = [item["detected_subtask_term_step"] for item in per_demo if item["detected_subtask_term_step"] is not None]
    auto_mp = int(np.median(mp_values)) if mp_values else None
    auto_term = int(np.median(term_values)) if term_values else None

    checks = []
    for name, manual, auto in (
        ("MP_end_step", manual_mp, auto_mp),
        ("subtask_term_step", manual_term, auto_term),
    ):
        if manual is None or auto is None:
            checks.append({"name": name, "manual": manual, "auto": auto, "diff": None, "status": "missing"})
            continue
        diff = abs(int(manual) - int(auto))
        checks.append(
            {
                "name": name,
                "manual": int(manual),
                "auto": int(auto),
                "diff": diff,
                "status": "ok" if diff <= max_manual_auto_diff else "warning",
            }
        )

    status = "ok" if checks and all(check["status"] == "ok" for check in checks) else "warning"
    return {
        "status": status,
        "action": action,
        "arm": arm,
        "object_ref": object_name,
        "manual": {
            "MP_end_step": manual_mp,
            "subtask_term_step": manual_term,
        },
        "auto": {
            "MP_end_step": auto_mp,
            "subtask_term_step": auto_term,
        },
        "checks": checks,
        "per_demo": per_demo,
    }


def apply_auto_boundaries_to_task_spec(
    task_spec: Any,
    source_dataset_path: str,
    task_name: str,
    mode: str = "compare_only",
    demo_keys: Optional[List[str]] = None,
    max_manual_auto_diff: int = DEFAULT_DETECTOR_CONFIG["max_manual_auto_diff"],
) -> Dict[str, Any]:
    """Compare or replace manual subtask boundaries in a ElogGen generation runtime task spec."""
    if mode not in VALID_AUTO_BOUNDARY_MODES:
        raise ValueError(f"Invalid auto boundary mode '{mode}'. Expected one of {VALID_AUTO_BOUNDARY_MODES}.")
    if mode == "off":
        return {"mode": mode, "status": "disabled", "updates": []}

    with h5py.File(source_dataset_path, "r") as dataset:
        all_demo_keys = _sorted_demo_keys(dataset["data"])
    demo_keys = all_demo_keys if demo_keys is None else demo_keys

    updates = []
    for phase_name, arm_key, subtask_name, subtask_spec in _iter_subtask_specs(task_spec):
        result = _detect_for_subtask(
            source_dataset_path=source_dataset_path,
            demo_keys=demo_keys,
            task_name=task_name,
            subtask_spec=subtask_spec,
            max_manual_auto_diff=max_manual_auto_diff,
        )
        result.update(
            {
                "phase": phase_name,
                "arm_key": arm_key,
                "subtask": subtask_name,
                "applied": False,
            }
        )
        if mode == "replace_manual" and result["status"] != "skipped":
            auto_mp = result["auto"]["MP_end_step"]
            auto_term = result["auto"]["subtask_term_step"]
            if auto_mp is not None:
                _set_item(subtask_spec, "MP_end_step", int(auto_mp))
            if auto_term is not None:
                _set_item(subtask_spec, "subtask_term_step", int(auto_term))
            result["applied"] = True
        updates.append(result)

    sync_records = []
    if mode == "replace_manual":
        sync_records = _sync_bimanual_phase_end_steps(task_spec=task_spec, updates=updates)

    num_warning = sum(1 for item in updates if item.get("status") == "warning")
    num_applied = sum(1 for item in updates if item.get("applied"))
    return {
        "mode": mode,
        "task": task_name,
        "source_dataset_path": source_dataset_path,
        "demo_keys": list(demo_keys),
        "summary": {
            "num_updates": len(updates),
            "num_applied": num_applied,
            "num_warning": num_warning,
            "num_skipped": sum(1 for item in updates if item.get("status") == "skipped"),
            "num_phase_syncs": len(sync_records),
        },
        "phase_syncs": sync_records,
        "updates": updates,
    }


def write_auto_boundary_report(report: Dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(_as_jsonable(report), file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")
    return str(path)
