"""V1 offline stage boundary detection for OpenArm manipulation demos."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

from eloggen.generation_runtime.context.actions import get_action_spec
from eloggen.generation_runtime.context.schema import (
    BoundaryCheck,
    BoundaryReport,
    BoundarySet,
    StageBoundary,
)


DEFAULT_TASK = "openarm_real_exp_1"
DEFAULT_ARM = "left"
DEFAULT_OBJECT = "object_1"
DEFAULT_ACTION = "pick"
DETECTOR_NAME = get_action_spec(DEFAULT_ACTION).detector

DEFAULT_DETECTOR_CONFIG = {
    "eef_object_dist_threshold": 0.09,
    "lift_z_threshold": 0.05,
    "max_manual_auto_diff": 50,
}

ARM_LABELS = {
    "left": {"zh": "左手", "en": "left hand"},
    "right": {"zh": "右手", "en": "right hand"},
}

OBJECT_LABEL_ZH_BY_TOKEN = {
    "apple": "苹果",
    "banana": "香蕉",
    "bottle": "瓶子",
    "bowl": "碗",
    "box": "盒子",
    "bag": "袋子",
    "candy": "糖果",
    "can": "罐子",
    "cup": "杯子",
    "drawer": "抽屉",
    "door": "门",
    "handle": "把手",
    "mug": "杯子",
    "plate": "盘子",
    "spoon": "勺子",
}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    def sort_key(name: str) -> Tuple[int, str]:
        if name.startswith("demo_"):
            try:
                return (int(name.split("_")[-1]), name)
            except ValueError:
                pass
        return (10**9, name)

    return sorted(data_group.keys(), key=sort_key)


def _first_index(condition: np.ndarray, default: Optional[int] = None) -> Optional[int]:
    indices = np.flatnonzero(condition)
    if indices.size == 0:
        return default
    return int(indices[0])


def _clip_stage(start: int, end: int, num_frames: int) -> Tuple[int, int]:
    start = max(0, min(int(start), num_frames - 1))
    end = max(start, min(int(end), num_frames - 1))
    return start, end


def _clean_object_id(object_name: Optional[str]) -> str:
    if object_name is None:
        return ""
    name = re.sub(r"_\d+$", "", str(object_name))
    return name.replace("_", " ").strip()


def _infer_object_label_en(object_name: Optional[str], fallback: str = "target object") -> str:
    return _clean_object_id(object_name) or fallback


def _infer_object_label_zh(object_name: Optional[str], fallback: str = "目标物体") -> str:
    tokens = _clean_object_id(object_name).lower().split()
    for token in reversed(tokens):
        if token in OBJECT_LABEL_ZH_BY_TOKEN:
            return OBJECT_LABEL_ZH_BY_TOKEN[token]
    return fallback


def _entity_labels(
    entity_name: Optional[str],
    config: Optional[Dict[str, Any]] = None,
    label_prefix: str = "object",
    fallback_zh: str = "目标物体",
    fallback_en: str = "target object",
) -> Tuple[str, str]:
    config = config or {}
    return (
        str(config.get(f"{label_prefix}_label_zh") or _infer_object_label_zh(entity_name, fallback=fallback_zh)),
        str(config.get(f"{label_prefix}_label_en") or _infer_object_label_en(entity_name, fallback=fallback_en)),
    )


def _object_labels(object_name: Optional[str], config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    return _entity_labels(object_name, config, label_prefix="object")


def _target_labels(target_name: Optional[str], config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    return _entity_labels(
        target_name,
        config,
        label_prefix="target",
        fallback_zh="目标位置",
        fallback_en="target location",
    )


def _resolve_target(action: str, config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    config = config or {}
    spec = get_action_spec(action)
    target_object = config.get("target_object")
    if not spec.supports_target and target_object is None:
        return None, None, None
    target_label_zh, target_label_en = _target_labels(target_object, config)
    return target_object, target_label_zh, target_label_en


def _format_stage_text(
    action: str,
    stage_type: str,
    active_arm: str,
    object_label_zh: str,
    object_label_en: str,
    target_label_zh: Optional[str],
    target_label_en: Optional[str],
) -> Dict[str, str]:
    arm_labels = ARM_LABELS.get(active_arm, {"zh": active_arm, "en": active_arm})
    values = {
        "arm_zh": arm_labels["zh"],
        "arm_en": arm_labels["en"],
        "object_zh": object_label_zh,
        "object_en": object_label_en,
        "target_zh": target_label_zh or "目标位置",
        "target_en": target_label_en or "target location",
    }
    stage_specs = {stage.stage_type: stage for stage in get_action_spec(action).stages}
    text = stage_specs[stage_type]
    return {
        "instruction_zh": text.instruction_zh.format(**values),
        "instruction_en": text.instruction_en.format(**values),
    }


def _make_stage(
    action: str,
    stage_type: str,
    substage_id: int,
    start: int,
    end: int,
    num_frames: int,
    active_arm: str,
    active_object: str,
    passive_object: str,
    object_label_zh: str,
    object_label_en: str,
    target_object: Optional[str] = None,
    target_label_zh: Optional[str] = None,
    target_label_en: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageBoundary:
    start, end = _clip_stage(start, end, num_frames)
    text = _format_stage_text(
        action,
        stage_type,
        active_arm,
        object_label_zh,
        object_label_en,
        target_label_zh,
        target_label_en,
    )
    return StageBoundary(
        action=action,
        stage_type=stage_type,
        substage_id=substage_id,
        start=start,
        end=end,
        active_arm=active_arm,
        active_object=active_object,
        passive_object=passive_object,
        object_label_zh=object_label_zh,
        object_label_en=object_label_en,
        target_object=target_object,
        target_label_zh=target_label_zh,
        target_label_en=target_label_en,
        instruction_zh=text["instruction_zh"],
        instruction_en=text["instruction_en"],
        metadata=metadata or {},
    )


def load_demo_signals(
    hdf5_path: str,
    demo_key: str,
    arm: str = DEFAULT_ARM,
    object_name: Optional[str] = DEFAULT_OBJECT,
) -> Dict[str, Any]:
    """Load the signals needed by the V1 pick boundary detector."""
    hdf5_path = str(hdf5_path)
    arm_index = 0 if arm == "left" else 1
    eef_slice = slice(0, 4) if arm == "left" else slice(4, 8)

    with h5py.File(hdf5_path, "r") as dataset:
        demo_group = dataset[f"data/{demo_key}"]
        datagen_info = demo_group["datagen_info"]
        eef_pose = datagen_info["eef_pose"][:]
        object_poses = datagen_info["object_poses"]
        if object_name is None:
            object_pose = None
        elif object_name not in object_poses:
            available = ", ".join(sorted(object_poses.keys()))
            raise KeyError(f"Object '{object_name}' not found in datagen_info/object_poses. Available: {available}")
        else:
            object_pose = object_poses[object_name][:]
        gripper_action = datagen_info["gripper_action"][:]
        actions = demo_group["action"][:] if "action" in demo_group else None

    eef_pose_arm = eef_pose[:, eef_slice, :]
    eef_pos = eef_pose_arm[:, :3, 3]
    if object_pose is None:
        object_pos = np.zeros_like(eef_pos)
        eef_object_dist = np.full(eef_pos.shape[0], np.inf, dtype=np.float32)
        object_z_delta = np.zeros(eef_pos.shape[0], dtype=np.float32)
    else:
        object_pos = object_pose[:, :3, 3]
        eef_object_dist = np.linalg.norm(eef_pos - object_pos, axis=1)
        object_z_delta = object_pos[:, 2] - object_pos[0, 2]
    gripper = gripper_action[:, arm_index]

    return {
        "hdf5_path": hdf5_path,
        "demo_key": demo_key,
        "arm": arm,
        "object_name": object_name,
        "num_frames": int(eef_pose.shape[0]),
        "eef_pose": eef_pose_arm,
        "eef_pos": eef_pos,
        "object_pose": object_pose,
        "object_pos": object_pos,
        "gripper_action": gripper,
        "all_gripper_action": gripper_action,
        "actions": actions,
        "eef_object_dist": eef_object_dist,
        "object_z_delta": object_z_delta,
    }


def detect_pick_boundaries(signals: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> BoundarySet:
    """Detect approach, grasp, lift, and retract substages for one pick-style demo."""
    action = "pick"
    spec = get_action_spec(action)
    cfg = dict(DEFAULT_DETECTOR_CONFIG)
    if config:
        cfg.update(config)

    num_frames = int(signals["num_frames"])
    active_arm = signals.get("arm", DEFAULT_ARM)
    active_object = str(cfg.get("active_object") or f"{active_arm}_gripper")
    passive_object = str(signals.get("object_name") or DEFAULT_OBJECT)
    object_label_zh, object_label_en = _object_labels(passive_object, cfg)
    target_object, target_label_zh, target_label_en = _resolve_target(action, cfg)
    dist = np.asarray(signals["eef_object_dist"])
    z_delta = np.asarray(signals["object_z_delta"])
    gripper = np.asarray(signals["gripper_action"])

    approach_end = _first_index(dist <= cfg["eef_object_dist_threshold"], default=int(np.argmin(dist)))
    grasp_start = _first_index(gripper < 0.0, default=approach_end)
    lift_start = _first_index(z_delta >= cfg["lift_z_threshold"], default=grasp_start)
    if lift_start is None:
        lift_start = grasp_start if grasp_start is not None else approach_end

    search_start = max(0, int(lift_start))
    lift_end = search_start + int(np.argmax(z_delta[search_start:])) if search_start < num_frames else num_frames - 1
    retract_start = min(num_frames - 1, int(lift_end) + 1)

    events = {
        "approach_start": 0,
        "approach_end": int(approach_end),
        "grasp_start": int(grasp_start),
        "lift_start": int(lift_start),
        "lift_end": int(lift_end),
        "subtask_end": int(lift_end),
        "retract_start": int(retract_start),
        "episode_end": num_frames - 1,
    }

    stages = [
        _make_stage(
            action,
            "approach",
            0,
            0,
            events["approach_end"],
            num_frames,
            active_arm,
            active_object,
            passive_object,
            object_label_zh,
            object_label_en,
            target_object=target_object,
            target_label_zh=target_label_zh,
            target_label_en=target_label_en,
            metadata={"end_event": "eef_object_dist_threshold"},
        ),
        _make_stage(
            action,
            "grasp",
            1,
            events["approach_end"] + 1,
            events["lift_start"] - 1,
            num_frames,
            active_arm,
            active_object,
            passive_object,
            object_label_zh,
            object_label_en,
            target_object=target_object,
            target_label_zh=target_label_zh,
            target_label_en=target_label_en,
            metadata={"grasp_start": events["grasp_start"]},
        ),
        _make_stage(
            action,
            "lift",
            2,
            events["lift_start"],
            events["lift_end"],
            num_frames,
            active_arm,
            active_object,
            passive_object,
            object_label_zh,
            object_label_en,
            target_object=target_object,
            target_label_zh=target_label_zh,
            target_label_en=target_label_en,
            metadata={"start_event": "object_z_delta_threshold"},
        ),
        _make_stage(
            action,
            "retract",
            3,
            events["retract_start"],
            events["episode_end"],
            num_frames,
            active_arm,
            active_object,
            passive_object,
            object_label_zh,
            object_label_en,
            target_object=target_object,
            target_label_zh=target_label_zh,
            target_label_en=target_label_en,
            metadata={"start_event": "post_lift"},
        ),
    ]

    metrics = {
        "eef_object_dist_threshold": cfg["eef_object_dist_threshold"],
        "lift_z_threshold": cfg["lift_z_threshold"],
        "min_eef_object_dist": float(np.min(dist)),
        "min_eef_object_dist_frame": int(np.argmin(dist)),
        "max_object_z_delta": float(np.max(z_delta)),
        "max_object_z_delta_frame": int(np.argmax(z_delta)),
        "gripper_close_first_frame": int(grasp_start),
    }

    return BoundarySet(
        demo_key=signals["demo_key"],
        task=DEFAULT_TASK,
        detector=spec.detector,
        action=action,
        num_frames=num_frames,
        active_arm=active_arm,
        active_object=active_object,
        passive_object=passive_object,
        object_label_zh=object_label_zh,
        object_label_en=object_label_en,
        stages=stages,
        events=events,
        metrics=metrics,
        target_object=target_object,
        target_label_zh=target_label_zh,
        target_label_en=target_label_en,
    )


def _split_span(start: int, end: int, num_parts: int) -> List[Tuple[int, int]]:
    if num_parts <= 0:
        return []
    if end < start:
        return [(start, start) for _ in range(num_parts)]
    edges = np.linspace(start, end + 1, num_parts + 1, dtype=int)
    spans = []
    for index in range(num_parts):
        span_start = int(edges[index])
        span_end = max(span_start, int(edges[index + 1]) - 1)
        spans.append((span_start, span_end))
    return spans


def detect_template_boundaries(
    signals: Dict[str, Any],
    action: str,
    config: Optional[Dict[str, Any]] = None,
) -> BoundarySet:
    """Build GenieSim-style substages from manual anchors for actions without a signal detector yet."""
    spec = get_action_spec(action)
    cfg = dict(DEFAULT_DETECTOR_CONFIG)
    if config:
        cfg.update(config)

    manual_subtask = cfg.get("manual_subtask") or {}
    num_frames = int(signals["num_frames"])
    active_arm = signals.get("arm", DEFAULT_ARM)
    active_object = str(cfg.get("active_object") or f"{active_arm}_gripper")
    passive_object = str(signals.get("object_name") or cfg.get("passive_object") or "")
    object_label_zh, object_label_en = _object_labels(passive_object, cfg)
    target_object, target_label_zh, target_label_en = _resolve_target(action, cfg)

    if len(spec.stages) == 1:
        stage = spec.stages[0]
        events = {
            f"{stage.stage_type}_start": 0,
            f"{stage.stage_type}_end": num_frames - 1,
            "subtask_end": num_frames - 1,
            "episode_end": num_frames - 1,
        }
        stages = [
            _make_stage(
                action,
                stage.stage_type,
                0,
                0,
                num_frames - 1,
                num_frames,
                active_arm,
                active_object,
                passive_object,
                object_label_zh,
                object_label_en,
                target_object=target_object,
                target_label_zh=target_label_zh,
                target_label_en=target_label_en,
                metadata={"boundary_source": "manual_config_template"},
            )
        ]
    else:
        default_approach_end = int((num_frames - 1) * 0.35)
        default_subtask_end = int((num_frames - 1) * 0.80)
        approach_end = manual_subtask.get("MP_end_step", default_approach_end)
        subtask_end = manual_subtask.get("subtask_term_step", default_subtask_end)
        approach_end, _ = _clip_stage(approach_end, approach_end, num_frames)
        subtask_end = max(approach_end, _clip_stage(subtask_end, subtask_end, num_frames)[0])

        has_retract = spec.stages[-1].stage_type == "retract"
        core_specs = spec.stages[1:-1] if has_retract else spec.stages[1:]
        core_spans = _split_span(approach_end + 1, subtask_end, len(core_specs))

        stages = [
            _make_stage(
                action,
                spec.stages[0].stage_type,
                0,
                0,
                approach_end,
                num_frames,
                active_arm,
                active_object,
                passive_object,
                object_label_zh,
                object_label_en,
                target_object=target_object,
                target_label_zh=target_label_zh,
                target_label_en=target_label_en,
                metadata={"boundary_source": "manual_config_template", "end_event": "MP_end_step"},
            )
        ]
        for offset, (stage, span) in enumerate(zip(core_specs, core_spans), start=1):
            stages.append(
                _make_stage(
                    action,
                    stage.stage_type,
                    offset,
                    span[0],
                    span[1],
                    num_frames,
                    active_arm,
                    active_object,
                    passive_object,
                    object_label_zh,
                    object_label_en,
                    target_object=target_object,
                    target_label_zh=target_label_zh,
                    target_label_en=target_label_en,
                    metadata={"boundary_source": "manual_config_template"},
                )
            )
        if has_retract:
            stages.append(
                _make_stage(
                    action,
                    spec.stages[-1].stage_type,
                    len(spec.stages) - 1,
                    min(num_frames - 1, subtask_end + 1),
                    num_frames - 1,
                    num_frames,
                    active_arm,
                    active_object,
                    passive_object,
                    object_label_zh,
                    object_label_en,
                    target_object=target_object,
                    target_label_zh=target_label_zh,
                    target_label_en=target_label_en,
                    metadata={"boundary_source": "manual_config_template", "start_event": "post_subtask"},
                )
            )

        events = {
            "approach_start": 0,
            "approach_end": int(approach_end),
            f"{action}_start": int(approach_end + 1),
            f"{action}_end": int(subtask_end),
            "subtask_end": int(subtask_end),
            "retract_start": int(min(num_frames - 1, subtask_end + 1)) if has_retract else num_frames - 1,
            "episode_end": num_frames - 1,
        }

    metrics = {
        "boundary_source": "manual_config_template",
        "manual_MP_end_step": manual_subtask.get("MP_end_step"),
        "manual_subtask_term_step": manual_subtask.get("subtask_term_step"),
    }

    return BoundarySet(
        demo_key=signals["demo_key"],
        task=DEFAULT_TASK,
        detector=spec.detector,
        action=action,
        num_frames=num_frames,
        active_arm=active_arm,
        active_object=active_object,
        passive_object=passive_object,
        object_label_zh=object_label_zh,
        object_label_en=object_label_en,
        stages=stages,
        events=events,
        metrics=metrics,
        target_object=target_object,
        target_label_zh=target_label_zh,
        target_label_en=target_label_en,
    )


def detect_action_boundaries(
    signals: Dict[str, Any],
    action: str = DEFAULT_ACTION,
    config: Optional[Dict[str, Any]] = None,
) -> BoundarySet:
    """Dispatch to the available detector for an action."""
    if action == "pick":
        return detect_pick_boundaries(signals, config=config)
    return detect_template_boundaries(signals, action=action, config=config)


def load_manual_steps(config_path: str, task_name: str = DEFAULT_TASK) -> Dict[str, Any]:
    """Load manual MP and subtask termination steps from a base config JSON."""
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)

    task_spec = config.get("task", {}).get("task_spec", {})
    phases = {}
    for phase_name, phase_config in task_spec.items():
        phase_result = {"type": phase_config.get("type"), "arms": {}}
        for arm_key in ("arm_left", "arm_right"):
            arm_entries = []
            for subtask_name, subtask_config in phase_config.get(arm_key, {}).items():
                arm_entries.append(
                    {
                        "subtask": subtask_name,
                        "arm": subtask_config.get("arm"),
                        "object_ref": subtask_config.get("object_ref"),
                        "target_ref": (
                            subtask_config.get("target_ref")
                            or subtask_config.get("target_object")
                            or subtask_config.get("placement_ref")
                            or subtask_config.get("container_ref")
                            or subtask_config.get("reference_obj")
                        ),
                        "action": subtask_config.get("action"),
                        "MP_end_step": subtask_config.get("MP_end_step"),
                        "subtask_term_step": subtask_config.get("subtask_term_step"),
                        "retract_type": subtask_config.get("retract_type"),
                    }
                )
            phase_result["arms"][arm_key] = arm_entries
        phases[phase_name] = phase_result

    return {
        "task": task_name,
        "config_path": str(config_path),
        "phases": phases,
    }


def _first_manual_arm_subtask(manual_steps: Dict[str, Any], arm: str = DEFAULT_ARM) -> Dict[str, Any]:
    phases = manual_steps.get("phases", {})
    arm_key = f"arm_{arm}"
    for phase_name in sorted(phases.keys()):
        entries = phases[phase_name].get("arms", {}).get(arm_key, [])
        if entries:
            return entries[0]
    return {}


def validate_boundaries(
    boundaries: BoundarySet,
    manual_steps: Dict[str, Any],
    max_diff: int = DEFAULT_DETECTOR_CONFIG["max_manual_auto_diff"],
) -> BoundaryReport:
    """Compare detected boundaries against manual task config steps."""
    manual_arm = _first_manual_arm_subtask(manual_steps, boundaries.active_arm)
    detected = boundaries.events
    checks: List[BoundaryCheck] = []

    comparisons = [
        ("MP_end_step", manual_arm.get("MP_end_step"), detected.get("approach_end")),
        ("subtask_term_step", manual_arm.get("subtask_term_step"), detected.get("subtask_end")),
    ]
    for name, manual, auto in comparisons:
        if manual is None or auto is None:
            checks.append(BoundaryCheck(name=name, manual=manual, detected=auto, diff=None, status="missing"))
            continue
        diff = abs(int(manual) - int(auto))
        checks.append(
            BoundaryCheck(
                name=name,
                manual=int(manual),
                detected=int(auto),
                diff=diff,
                status="ok" if diff <= max_diff else "warning",
            )
        )

    status = "ok" if all(check.status == "ok" for check in checks) else "warning"
    return BoundaryReport(
        demo_key=boundaries.demo_key,
        task=boundaries.task,
        detector=boundaries.detector,
        status=status,
        checks=checks,
        manual_steps=manual_steps,
        detected_events=boundaries.events,
        max_diff=int(max_diff),
    )


def write_boundary_outputs(
    boundary_sets: Iterable[BoundarySet],
    reports: Iterable[BoundaryReport],
    output_dir: str,
) -> Dict[str, str]:
    """Write stage boundaries and validation reports to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    boundary_sets = list(boundary_sets)
    reports = list(reports)
    stage_path = output_path / "stage_boundaries.json"
    report_path = output_path / "boundary_report.json"

    stage_payload = {
        "task": boundary_sets[0].task if boundary_sets else None,
        "detector": boundary_sets[0].detector if boundary_sets else DETECTOR_NAME,
        "boundary_sets": {item.demo_key: item.to_dict() for item in boundary_sets},
    }
    report_payload = {
        "task": reports[0].task if reports else None,
        "detector": reports[0].detector if reports else DETECTOR_NAME,
        "summary": {
            "num_demos": len(reports),
            "num_ok": sum(1 for report in reports if report.status == "ok"),
            "num_warning": sum(1 for report in reports if report.status != "ok"),
        },
        "reports": {item.demo_key: item.to_dict() for item in reports},
    }

    with open(stage_path, "w", encoding="utf-8") as file_obj:
        json.dump(_as_jsonable(stage_payload), file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")
    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(_as_jsonable(report_payload), file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")

    return {
        "stage_boundaries": str(stage_path),
        "boundary_report": str(report_path),
    }


def annotate_stage_boundaries(
    input_path: str,
    task_name: str,
    output_dir: str,
    demo_key: str = "all",
    mode: str = "validate",
    action: str = DEFAULT_ACTION,
    config_path: Optional[str] = None,
    arm: str = DEFAULT_ARM,
    object_name: Optional[str] = None,
    object_label_zh: Optional[str] = None,
    object_label_en: Optional[str] = None,
    active_object: Optional[str] = None,
    target_name: Optional[str] = None,
    target_label_zh: Optional[str] = None,
    target_label_en: Optional[str] = None,
    detector_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run V1 offline boundary annotation for one or all demos in a processed source HDF5."""
    if mode != "validate":
        raise ValueError("V1 only supports mode='validate'. Boundary replacement is a V3 feature.")
    action_spec = get_action_spec(action)

    if config_path is None:
        config_path = str(_repo_root() / "generation_runtime" / "datasets" / "base_configs" / f"{task_name}.json")
    manual_steps = load_manual_steps(config_path=config_path, task_name=task_name)
    manual_subtask = _first_manual_arm_subtask(manual_steps, arm)
    if object_name is None:
        object_name = manual_subtask.get("object_ref")
        if object_name is None and action_spec.requires_object:
            object_name = DEFAULT_OBJECT
    if target_name is None:
        target_name = manual_subtask.get("target_ref")

    detector_config = dict(detector_config or {})
    detector_config["manual_subtask"] = manual_subtask
    if object_label_zh is not None:
        detector_config["object_label_zh"] = object_label_zh
    if object_label_en is not None:
        detector_config["object_label_en"] = object_label_en
    if active_object is not None:
        detector_config["active_object"] = active_object
    if target_name is not None:
        detector_config["target_object"] = target_name
    if target_label_zh is not None:
        detector_config["target_label_zh"] = target_label_zh
    if target_label_en is not None:
        detector_config["target_label_en"] = target_label_en

    with h5py.File(input_path, "r") as dataset:
        all_demo_keys = _sorted_demo_keys(dataset["data"])
    demo_keys = all_demo_keys if demo_key == "all" else [demo_key]

    boundary_sets = []
    reports = []
    for key in demo_keys:
        signals = load_demo_signals(input_path, key, arm=arm, object_name=object_name)
        boundaries = detect_action_boundaries(signals, action=action, config=detector_config)
        boundaries.task = task_name
        report = validate_boundaries(
            boundaries,
            manual_steps,
            max_diff=detector_config.get(
                "max_manual_auto_diff",
                DEFAULT_DETECTOR_CONFIG["max_manual_auto_diff"],
            ),
        )
        boundary_sets.append(boundaries)
        reports.append(report)

    output_files = write_boundary_outputs(boundary_sets, reports, output_dir)
    return {
        "input": str(input_path),
        "task": task_name,
        "action": action,
        "mode": mode,
        "demo_keys": demo_keys,
        "output_files": output_files,
        "summary": {
            "num_demos": len(reports),
            "num_ok": sum(1 for report in reports if report.status == "ok"),
            "num_warning": sum(1 for report in reports if report.status != "ok"),
        },
    }
