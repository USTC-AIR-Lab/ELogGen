"""Infer hierarchical task / subtask / phase semantics from ElogGen generation runtime task specs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _get_item(container: Any, key: str, default: Any = None) -> Any:
    try:
        return container[key]
    except Exception:
        return getattr(container, key, default)


def _object_label(object_name: Optional[str]) -> str:
    if object_name is None:
        return "object"
    name = re.sub(r"_\d+$", "", str(object_name))
    return name.replace("_", " ").strip() or str(object_name)


def _label_or_object(label: Optional[str], object_name: Optional[str]) -> str:
    return str(label).strip() if label else _object_label(object_name)


@dataclass
class PhaseSemantic:
    phase_index: int
    subtask_index: int
    active_arm: Optional[str]
    phase_action: str
    phase_active_object: Optional[str]
    phase_target_object: Optional[str]
    phase_reference_object: Optional[str]
    phase_attached_object: Optional[str]
    subtask_id: int
    subtask_type: str
    subtask_object: Optional[str]
    subtask_target: Optional[str]
    phase_instruction_zh: str
    phase_instruction_en: str
    subtask_instruction_zh: str
    subtask_instruction_en: str
    inference_source: str = "auto_task_spec"
    phase_active_object_label_zh: Optional[str] = None
    phase_active_object_label_en: Optional[str] = None
    phase_target_object_label_zh: Optional[str] = None
    phase_target_object_label_en: Optional[str] = None
    subtask_object_label_zh: Optional[str] = None
    subtask_object_label_en: Optional[str] = None
    subtask_target_label_zh: Optional[str] = None
    subtask_target_label_en: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _active_arm_from_phase_specs(phase_specs: Any, subtask_index: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    candidates = []
    for arm_index, arm_name in enumerate(("left", "right")):
        try:
            arm_specs = phase_specs[arm_index]
        except Exception:
            continue
        if len(arm_specs) == 0:
            continue
        local_index = min(max(int(subtask_index), 0), len(arm_specs) - 1)
        spec = dict(arm_specs[local_index])
        if spec.get("object_ref") is not None or spec.get("attached_obj") is not None:
            candidates.append((arm_name, spec))

    if candidates:
        return candidates[0]

    for arm_index, arm_name in enumerate(("left", "right")):
        try:
            arm_specs = phase_specs[arm_index]
        except Exception:
            continue
        if len(arm_specs) == 0:
            continue
        local_index = min(max(int(subtask_index), 0), len(arm_specs) - 1)
        return arm_name, dict(arm_specs[local_index])

    return None, None


def _infer_phase_action(spec: Dict[str, Any]) -> str:
    explicit = spec.get("action") or spec.get("phase_action")
    if explicit is not None:
        return str(explicit)
    if spec.get("attached_obj") is not None and spec.get("object_ref") is not None:
        return "place"
    if spec.get("object_ref") is not None:
        return "pick"
    return "reset"


def _phase_instruction_zh(
    action: str,
    obj: Optional[str],
    target: Optional[str],
    obj_label_zh: Optional[str] = None,
    target_label_zh: Optional[str] = None,
) -> str:
    obj_label = _label_or_object(obj_label_zh, obj)
    target_label = _label_or_object(target_label_zh, target)
    if action == "pick":
        return f"抓起{obj_label}"
    if action == "open":
        return f"打开{obj_label}"
    if action == "place":
        return f"把{obj_label}放到{target_label}"
    if action == "insert":
        return f"把{obj_label}放入{target_label}"
    if action == "reset":
        return "回到初始位姿"
    return f"执行{action}"


def _phase_instruction_en(
    action: str,
    obj: Optional[str],
    target: Optional[str],
    obj_label_en: Optional[str] = None,
    target_label_en: Optional[str] = None,
) -> str:
    obj_label = _label_or_object(obj_label_en, obj)
    target_label = _label_or_object(target_label_en, target)
    if action == "pick":
        return f"Pick up {obj_label}"
    if action == "open":
        return f"Open the {obj_label}"
    if action == "place":
        return f"Place {obj_label} at {target_label}"
    if action == "insert":
        return f"Insert {obj_label} into {target_label}"
    if action == "reset":
        return "Return to the initial pose"
    return f"Execute {action}"


def _subtask_instruction_zh(
    subtask_type: str,
    obj: Optional[str],
    target: Optional[str],
    obj_label_zh: Optional[str] = None,
    target_label_zh: Optional[str] = None,
) -> str:
    obj_label = _label_or_object(obj_label_zh, obj)
    target_label = _label_or_object(target_label_zh, target)
    if subtask_type == "transfer" and target is not None:
        return f"把{obj_label}放到{target_label}"
    if subtask_type == "pick":
        return f"抓起{obj_label}"
    if subtask_type == "place" and target is not None:
        return f"放置{obj_label}到{target_label}"
    return f"操作{obj_label}"


def _subtask_instruction_en(
    subtask_type: str,
    obj: Optional[str],
    target: Optional[str],
    obj_label_en: Optional[str] = None,
    target_label_en: Optional[str] = None,
) -> str:
    obj_label = _label_or_object(obj_label_en, obj)
    target_label = _label_or_object(target_label_en, target)
    if subtask_type == "transfer" and target is not None:
        return f"Move {obj_label} to {target_label}"
    if subtask_type == "pick":
        return f"Pick up {obj_label}"
    if subtask_type == "place" and target is not None:
        return f"Place {obj_label} at {target_label}"
    return f"Manipulate {obj_label}"


def _num_subtasks_for_phase(phase_specs: Any) -> int:
    lengths = []
    for arm_index in range(2):
        try:
            lengths.append(len(phase_specs[arm_index]))
        except Exception:
            pass
    return max(lengths) if lengths else 0


def _initial_semantics(task_spec: Any) -> List[PhaseSemantic]:
    semantics = []
    next_subtask_id = 0
    open_pick_by_object: Dict[str, int] = {}

    for phase_index in range(len(task_spec)):
        phase_specs = task_spec[phase_index]
        for subtask_index in range(_num_subtasks_for_phase(phase_specs)):
            active_arm, spec = _active_arm_from_phase_specs(phase_specs, subtask_index)
            if spec is None:
                continue

            object_ref = spec.get("object_ref")
            attached_obj = spec.get("attached_obj")
            action = _infer_phase_action(spec)
            active_object = spec.get("active_object")
            if active_object is None:
                active_object = attached_obj if action in {"place", "insert"} and attached_obj is not None else object_ref
            target_object = (
                spec.get("target_ref")
                or spec.get("target_object")
                or spec.get("placement_ref")
                or spec.get("container_ref")
            )
            if target_object is None and action in {"place", "insert"}:
                target_object = object_ref
            active_object_label_zh = spec.get("object_label_zh")
            active_object_label_en = spec.get("object_label_en")
            target_object_label_zh = spec.get("target_label_zh")
            target_object_label_en = spec.get("target_label_en")

            explicit_subtask_id = spec.get("subtask_id")
            if explicit_subtask_id is not None:
                subtask_id = int(explicit_subtask_id)
            elif action in {"place", "insert"} and active_object in open_pick_by_object:
                subtask_id = open_pick_by_object.pop(str(active_object))
            else:
                subtask_id = next_subtask_id
                next_subtask_id += 1

            if action == "pick" and active_object is not None:
                open_pick_by_object[str(active_object)] = subtask_id

            subtask_type = str(spec.get("subtask_type") or ("place" if action in {"place", "insert"} else action))
            semantics.append(
                PhaseSemantic(
                    phase_index=phase_index,
                    subtask_index=subtask_index,
                    active_arm=active_arm,
                    phase_action=action,
                    phase_active_object=active_object,
                    phase_target_object=target_object,
                    phase_reference_object=object_ref,
                    phase_attached_object=attached_obj,
                    subtask_id=subtask_id,
                    subtask_type=subtask_type,
                    subtask_object=active_object,
                    subtask_target=target_object,
                    phase_instruction_zh=str(
                        spec.get("phase_instruction_zh")
                        or _phase_instruction_zh(
                            action,
                            active_object,
                            target_object,
                            active_object_label_zh,
                            target_object_label_zh,
                        )
                    ),
                    phase_instruction_en=str(
                        spec.get("phase_instruction_en")
                        or _phase_instruction_en(
                            action,
                            active_object,
                            target_object,
                            active_object_label_en,
                            target_object_label_en,
                        )
                    ),
                    subtask_instruction_zh="",
                    subtask_instruction_en="",
                    inference_source="explicit_task_spec" if explicit_subtask_id is not None or spec.get("action") else "auto_task_spec",
                    phase_active_object_label_zh=active_object_label_zh,
                    phase_active_object_label_en=active_object_label_en,
                    phase_target_object_label_zh=target_object_label_zh,
                    phase_target_object_label_en=target_object_label_en,
                )
            )

    return semantics


def build_semantic_plan(task_spec: Any) -> Dict[Tuple[int, int], PhaseSemantic]:
    """Infer phase and higher-level subtask semantics for a bimanual task spec."""
    semantics = _initial_semantics(task_spec)
    grouped: Dict[int, List[PhaseSemantic]] = {}
    for semantic in semantics:
        grouped.setdefault(semantic.subtask_id, []).append(semantic)

    for subtask_id, group in grouped.items():
        object_candidates = [item.phase_active_object for item in group if item.phase_active_object is not None]
        target_candidates = [item.phase_target_object for item in group if item.phase_target_object is not None]
        object_label_zh_candidates = [
            item.phase_active_object_label_zh for item in group if item.phase_active_object_label_zh is not None
        ]
        object_label_en_candidates = [
            item.phase_active_object_label_en for item in group if item.phase_active_object_label_en is not None
        ]
        target_label_zh_candidates = [
            item.phase_target_object_label_zh for item in group if item.phase_target_object_label_zh is not None
        ]
        target_label_en_candidates = [
            item.phase_target_object_label_en for item in group if item.phase_target_object_label_en is not None
        ]
        subtask_object = object_candidates[0] if object_candidates else None
        subtask_target = target_candidates[-1] if target_candidates else None
        subtask_object_label_zh = object_label_zh_candidates[0] if object_label_zh_candidates else None
        subtask_object_label_en = object_label_en_candidates[0] if object_label_en_candidates else None
        subtask_target_label_zh = target_label_zh_candidates[-1] if target_label_zh_candidates else None
        subtask_target_label_en = target_label_en_candidates[-1] if target_label_en_candidates else None
        actions = {item.phase_action for item in group}
        if "pick" in actions and ({"place", "insert"} & actions):
            subtask_type = "transfer"
        elif "place" in actions:
            subtask_type = "place"
        elif "pick" in actions:
            subtask_type = "pick"
        else:
            subtask_type = group[0].subtask_type

        for item in group:
            item.subtask_type = subtask_type
            item.subtask_object = subtask_object
            item.subtask_target = subtask_target
            item.subtask_object_label_zh = subtask_object_label_zh
            item.subtask_object_label_en = subtask_object_label_en
            item.subtask_target_label_zh = subtask_target_label_zh
            item.subtask_target_label_en = subtask_target_label_en
            item.subtask_instruction_zh = _subtask_instruction_zh(
                subtask_type,
                subtask_object,
                subtask_target,
                subtask_object_label_zh,
                subtask_target_label_zh,
            )
            item.subtask_instruction_en = _subtask_instruction_en(
                subtask_type,
                subtask_object,
                subtask_target,
                subtask_object_label_en,
                subtask_target_label_en,
            )

    return {(item.phase_index, item.subtask_index): item for item in semantics}


def apply_phase_semantic_to_task_spec(task_spec: Dict[str, Any], semantic: Optional[PhaseSemantic]) -> Dict[str, Any]:
    """Merge inferred semantic fields into a local task spec for boundary detection."""
    spec = dict(task_spec or {})
    if semantic is None:
        return spec

    spec.setdefault("action", semantic.phase_action)
    spec.setdefault("active_object", semantic.phase_active_object)
    spec.setdefault("target_ref", semantic.phase_target_object)
    spec.setdefault("target_object", semantic.phase_target_object)
    if semantic.phase_active_object_label_zh is not None:
        spec.setdefault("object_label_zh", semantic.phase_active_object_label_zh)
    if semantic.phase_active_object_label_en is not None:
        spec.setdefault("object_label_en", semantic.phase_active_object_label_en)
    if semantic.phase_target_object_label_zh is not None:
        spec.setdefault("target_label_zh", semantic.phase_target_object_label_zh)
    if semantic.phase_target_object_label_en is not None:
        spec.setdefault("target_label_en", semantic.phase_target_object_label_en)
    spec.setdefault("phase_instruction_zh", semantic.phase_instruction_zh)
    spec.setdefault("phase_instruction_en", semantic.phase_instruction_en)
    return spec


def attach_semantics_to_frame_contexts(
    frame_contexts: Iterable[Dict[str, Any]],
    semantic: Optional[PhaseSemantic],
) -> List[Dict[str, Any]]:
    """Attach hierarchical phase/subtask fields to per-frame contexts."""
    contexts = [dict(context) for context in frame_contexts]
    if semantic is None:
        for context in contexts:
            context.setdefault("skill_type", context.get("stage_type"))
        return contexts

    payload = semantic.to_dict()
    for context in contexts:
        context.update(payload)
        context["action"] = semantic.phase_action
        context["skill_type"] = context.get("stage_type")
        context["phase_action"] = semantic.phase_action
        context["phase_reference_object"] = semantic.phase_reference_object
        context["phase_attached_object"] = semantic.phase_attached_object
        context["subtask_id"] = semantic.subtask_id
        context["subtask_index_in_task"] = semantic.subtask_id
        context["subtask_type"] = semantic.subtask_type
        context["subtask_object"] = semantic.subtask_object
        context["subtask_target"] = semantic.subtask_target
        if semantic.phase_active_object_label_zh is not None:
            context["object_label_zh"] = semantic.phase_active_object_label_zh
        if semantic.phase_active_object_label_en is not None:
            context["object_label_en"] = semantic.phase_active_object_label_en
            context["active_object_name"] = semantic.phase_active_object_label_en
            context["phase_active_object_name"] = semantic.phase_active_object_label_en
        if semantic.phase_target_object_label_zh is not None:
            context["target_label_zh"] = semantic.phase_target_object_label_zh
        if semantic.phase_target_object_label_en is not None:
            context["target_label_en"] = semantic.phase_target_object_label_en
            context["target_object_name"] = semantic.phase_target_object_label_en
            context["phase_target_object_name"] = semantic.phase_target_object_label_en
        if semantic.subtask_object_label_en is not None:
            context["subtask_object_name"] = semantic.subtask_object_label_en
        if semantic.subtask_target_label_en is not None:
            context["subtask_target_name"] = semantic.subtask_target_label_en
    return contexts
