"""Task graph loading and validation."""

from __future__ import annotations

from .model import TaskGraph
from eloggen.datasets import resolve_dataset_path
from eloggen.pipeline.io import load_structured_file


def load_task_graph(path: str) -> TaskGraph:
    path = resolve_dataset_path(path) or path
    data = load_structured_file(path)
    return TaskGraph.from_dict(data)


def validate_task_graph(task_graph: TaskGraph) -> None:
    object_names = set(task_graph.object_names())
    target_names = set(task_graph.target_names())
    effector_names = set(task_graph.effector_names())
    for binding in task_graph.bindings:
        if binding.object not in object_names:
            raise ValueError(f"Binding references unknown object: {binding.object}")
        if binding.target is not None and binding.target not in target_names:
            raise ValueError(f"Binding references unknown target: {binding.target}")
        for effector in binding.allowed_effectors:
            if effector not in effector_names:
                raise ValueError(f"Binding references unknown effector: {effector}")

