"""Action and substage templates for context boundary injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class StageSpec:
    stage_type: str
    instruction_zh: str
    instruction_en: str


@dataclass(frozen=True)
class ActionSpec:
    action: str
    detector: str
    stages: Tuple[StageSpec, ...]
    requires_object: bool = True
    supports_target: bool = False


ACTION_SPECS: Dict[str, ActionSpec] = {
    "open": ActionSpec(
        action="open",
        detector="articulated_open_template_v1",
        stages=(
            StageSpec("approach_handle", "{arm_zh}靠近{object_zh}把手", "Move the {arm_en} toward the {object_en} handle"),
            StageSpec("grasp_handle", "{arm_zh}抓住{object_zh}把手", "Grasp the {object_en} handle with the {arm_en}"),
            StageSpec("pull_open", "{arm_zh}拉开{object_zh}", "Pull the {object_en} open with the {arm_en}"),
            StageSpec("release_handle", "{arm_zh}松开{object_zh}把手", "Release the {object_en} handle"),
            StageSpec("retract", "{arm_zh}离开{object_zh}", "Retract the {arm_en} from the {object_en}"),
        ),
    ),
    "pick": ActionSpec(
        action="pick",
        detector="pick_v1",
        stages=(
            StageSpec("approach", "{arm_zh}靠近{object_zh}", "Move the {arm_en} toward the {object_en}"),
            StageSpec("grasp", "{arm_zh}闭合夹爪并抓取{object_zh}", "Close the {arm_en} gripper and grasp the {object_en}"),
            StageSpec("lift", "{arm_zh}拿起{object_zh}", "Lift the {object_en} with the {arm_en}"),
            StageSpec("retract", "{arm_zh}带着{object_zh}收回", "Retract the {arm_en} while holding the {object_en}"),
        ),
    ),
    "place": ActionSpec(
        action="place",
        detector="place_template_v1",
        supports_target=True,
        stages=(
            StageSpec("approach_place", "{arm_zh}带着{object_zh}靠近{target_zh}", "Move the {object_en} toward the {target_en}"),
            StageSpec("place", "{arm_zh}将{object_zh}放到{target_zh}", "Place the {object_en} at the {target_en}"),
            StageSpec("release", "{arm_zh}打开夹爪释放{object_zh}", "Open the {arm_en} gripper and release the {object_en}"),
            StageSpec("retract", "{arm_zh}离开{object_zh}", "Move the {arm_en} away from the {object_en}"),
        ),
    ),
    "insert": ActionSpec(
        action="insert",
        detector="insert_template_v1",
        supports_target=True,
        stages=(
            StageSpec("approach_insert", "{arm_zh}带着{object_zh}靠近{target_zh}", "Move the {object_en} toward the {target_en}"),
            StageSpec("align", "{arm_zh}将{object_zh}对准{target_zh}", "Align the {object_en} with the {target_en}"),
            StageSpec("insert", "{arm_zh}把{object_zh}插入{target_zh}", "Insert the {object_en} into the {target_en}"),
            StageSpec("release", "{arm_zh}释放{object_zh}", "Release the {object_en}"),
            StageSpec("retract", "{arm_zh}收回", "Retract the {arm_en}"),
        ),
    ),
    "rotate": ActionSpec(
        action="rotate",
        detector="rotate_template_v1",
        stages=(
            StageSpec("approach", "{arm_zh}靠近{object_zh}", "Move the {arm_en} toward the {object_en}"),
            StageSpec("grasp", "{arm_zh}抓住{object_zh}", "Grasp the {object_en} with the {arm_en}"),
            StageSpec("rotate", "{arm_zh}旋转{object_zh}", "Rotate the {object_en} with the {arm_en}"),
            StageSpec("release", "{arm_zh}释放{object_zh}", "Release the {object_en}"),
            StageSpec("retract", "{arm_zh}收回", "Retract the {arm_en}"),
        ),
    ),
    "reset": ActionSpec(
        action="reset",
        detector="reset_template_v1",
        stages=(
            StageSpec("reset", "{arm_zh}回到初始位姿", "Reset the {arm_en} to the initial pose"),
        ),
        requires_object=False,
    ),
}


def get_action_spec(action: str) -> ActionSpec:
    try:
        return ACTION_SPECS[action]
    except KeyError as exc:
        supported = ", ".join(sorted(ACTION_SPECS))
        raise ValueError(f"Unsupported action '{action}'. Supported actions: {supported}") from exc


def supported_actions() -> Tuple[str, ...]:
    return tuple(sorted(ACTION_SPECS))
