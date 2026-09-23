"""Compile reusable articulated-storage task graphs from task specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledStorageTaskGraph:
    graph: dict
    node_to_phase: dict[str, int]
    node_metadata: dict[str, dict]
    warnings: tuple[str, ...] = ()


def _ordered_phases(task_spec: dict) -> list[dict]:
    def phase_number(item):
        key = item[0]
        try:
            return int(str(key).rsplit("_", 1)[-1])
        except ValueError:
            return 0

    return [phase for _, phase in sorted(task_spec.items(), key=phase_number)]


def _active_subtask(phase: dict) -> dict:
    active_effector = phase.get("active_effector")
    arm_key = f"arm_{active_effector}" if active_effector else None
    arm_specs = phase.get(arm_key, {}) if arm_key else {}
    candidates = list(arm_specs.values())
    if not candidates:
        candidates = [
            subtask
            for key, arm in phase.items()
            if str(key).startswith("arm_") and isinstance(arm, dict)
            for subtask in arm.values()
            if isinstance(subtask, dict) and subtask.get("action")
        ]
    active = [candidate for candidate in candidates if candidate.get("action")]
    if len(active) != 1:
        raise ValueError(
            f"phase {phase.get('node_id')!r} must contain exactly one active action subtask, found {len(active)}"
        )
    return active[0]


def compile_storage_task_graph(task_spec: dict) -> CompiledStorageTaskGraph:
    """Infer dependencies and exclusive gripper events for storage tasks."""
    nodes = []
    node_to_phase = {}
    metadata = {}
    picks = {}
    places = {}
    open_nodes_by_target = {}
    resource_events = {}

    for phase_index, phase in enumerate(_ordered_phases(task_spec)):
        node = phase.get("node_id")
        if not node or node in node_to_phase:
            raise ValueError(f"phase {phase_index} has missing or duplicate node_id: {node!r}")
        active = _active_subtask(phase)
        action = str(active["action"])
        effector = str(phase.get("active_effector") or active.get("arm") or "")
        if not effector:
            raise ValueError(f"node {node!r} has no active effector")

        nodes.append(node)
        node_to_phase[node] = phase_index
        item = active.get("attached_obj") if action in {"insert", "place"} else active.get("object_ref")
        target = None
        if action == "open":
            target = (phase.get("articulated_action") or {}).get("target_id")
            if not target:
                raise ValueError(f"open node {node!r} has no articulated target")
            open_nodes_by_target.setdefault(str(target), []).append(node)
        elif action == "pick":
            if not item or item in picks:
                raise ValueError(f"pick node {node!r} has missing or duplicate item: {item!r}")
            picks[str(item)] = node
            resource_events[node] = [
                {"resource": f"{effector}_gripper", "operation": "acquire", "token": str(item)}
            ]
        elif action in {"insert", "place"}:
            if not item or item in places:
                raise ValueError(f"place node {node!r} has missing or duplicate item: {item!r}")
            target = (phase.get("placement_target") or {}).get("articulated_target_id")
            if not target:
                raise ValueError(f"place node {node!r} has no articulated target")
            places[str(item)] = (node, str(target))
            resource_events[node] = [
                {"resource": f"{effector}_gripper", "operation": "release", "token": str(item)}
            ]
        else:
            raise ValueError(f"unsupported storage action {action!r} in node {node!r}")
        metadata[node] = {
            "phase_index": phase_index,
            "action": action,
            "active_effector": effector,
            "object_ref": item,
            "articulated_target_id": target,
        }

    if set(picks) != set(places):
        raise ValueError(
            f"pick/place object mismatch: picks={sorted(picks)}, places={sorted(places)}"
        )

    edges = []
    for item, pick_node in picks.items():
        place_node, target = places[item]
        edges.append([pick_node, place_node])
        open_nodes = open_nodes_by_target.get(target, [])
        if len(open_nodes) != 1:
            raise ValueError(
                f"item {item!r} target {target!r} requires exactly one open node, found {open_nodes}"
            )
        edges.append([open_nodes[0], place_node])

    resource_constraints = {
        item: [picks[item], places[item][0]]
        for item in picks
    }
    for target, open_nodes in open_nodes_by_target.items():
        resource_constraints[f"{target}_handle"] = list(open_nodes)
    for node, events in resource_events.items():
        for event in events:
            resource_constraints.setdefault(event["resource"], []).append(node)

    return CompiledStorageTaskGraph(
        graph={
            "nodes": nodes,
            "edges": edges,
            "atomic_chains": [],
            "resource_constraints": resource_constraints,
            "resource_events": resource_events,
        },
        node_to_phase=node_to_phase,
        node_metadata=metadata,
    )
