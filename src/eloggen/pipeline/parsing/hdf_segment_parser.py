"""Parse ElogGen generation runtime HDF demonstrations into paper-level interaction segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from eloggen.datasets import resolve_dataset_path
from ..replay_annotation import ReplayAnnotation, replay_annotation_from_task_graph
from eloggen.planning.model import (
    InteractionSegment,
    Predicate,
    SemanticRole,
    SourceTrajectorySlice,
    TemporalBoundary,
)
from ..source_parser import task_graph_from_generation_config
from .boundary_detectors import detect_gripper_transitions, resolve_unit_boundaries
from ..io import write_json


@dataclass
class ParsedSegments:
    segments: List[InteractionSegment]
    source_manifest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "source_manifest": dict(self.source_manifest),
        }

    def write(self, output_dir: str) -> None:
        output = Path(output_dir)
        write_json(output / "segments.json", {"segments": [item.to_dict() for item in self.segments]})
        write_json(output / "source_manifest.json", self.source_manifest)


class HDFSegmentParser:
    def __init__(self, config_path: str, hdf5_path: str, demo_id: str = "demo_0") -> None:
        self.config_path = resolve_dataset_path(config_path) or str(config_path)
        self.hdf5_path = resolve_dataset_path(hdf5_path) or str(hdf5_path)
        self.demo_id = demo_id

    def parse(self, annotation: Optional[ReplayAnnotation] = None) -> ParsedSegments:
        import h5py  # type: ignore

        if not Path(self.hdf5_path).is_file():
            raise FileNotFoundError(self.hdf5_path)
        graph = task_graph_from_generation_config(self.config_path, self.hdf5_path)
        annotation = annotation or replay_annotation_from_task_graph(graph, self.demo_id)
        ranges = resolve_unit_boundaries(graph.semantic_units, annotation)

        with h5py.File(self.hdf5_path, "r") as hdf:
            demo_path = f"data/{self.demo_id}"
            if demo_path not in hdf:
                raise KeyError(f"Demo group {demo_path!r} not found in {self.hdf5_path}")
            demo = hdf[demo_path]
            dataset_keys: List[str] = []
            shapes: Dict[str, List[int]] = {}

            def collect(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    dataset_keys.append(name)
                    shapes[name] = list(obj.shape)

            demo.visititems(collect)
            frame_counts = [shape[0] for shape in shapes.values() if shape]
            num_frames = min(frame_counts) if frame_counts else 0
            gripper_path = "datagen_info/gripper_action"
            transitions = (
                detect_gripper_transitions(demo[gripper_path][...]) if gripper_path in demo else {}
            )

        segments: List[InteractionSegment] = []
        for unit in graph.semantic_units:
            if unit.id not in ranges:
                continue
            start, end, boundary_source = ranges[unit.id]
            if start < 0 or end > num_frames or start >= end:
                raise ValueError(
                    f"Segment {unit.id} has invalid range [{start}, {end}) for {num_frames} frames"
                )
            roles = [SemanticRole("manipulated_object", str(unit.object))]
            if unit.target is not None:
                roles.append(SemanticRole("target", unit.target))
            preconditions, effects = self._predicates(unit.action, str(unit.object), unit.target, str(unit.effector))
            effector_column = {"left": 0, "right": 1}.get(str(unit.effector))
            transition_frames = transitions.get(effector_column, []) if effector_column is not None else []
            in_range_transitions = [frame for frame in transition_frames if start <= frame < end]
            segments.append(
                InteractionSegment(
                    id=unit.id,
                    alpha=unit.action,
                    roles=roles,
                    effectors=[str(unit.effector)],
                    boundary=TemporalBoundary(start, end, self.demo_id),
                    source_slice=SourceTrajectorySlice(
                        hdf5_path=str(Path(self.hdf5_path).resolve()),
                        demo_key=self.demo_id,
                        start_frame=start,
                        end_frame=end,
                        dataset_keys=dataset_keys,
                        metadata={"num_frames": end - start},
                    ),
                    preconditions=preconditions,
                    effects=effects,
                    metadata={
                        "phase_key": unit.metadata.get("phase_key"),
                        "boundary_source": boundary_source,
                        "source_semantic_unit": unit.id,
                        "gripper_transition_frames": in_range_transitions,
                        "boundary_verified_by_gripper": bool(in_range_transitions),
                    },
                )
            )

        manifest = {
            "hdf5_path": str(Path(self.hdf5_path).resolve()),
            "config_path": str(Path(self.config_path).resolve()),
            "demo_id": self.demo_id,
            "num_frames": num_frames,
            "dataset_keys": dataset_keys,
            "dataset_shapes": shapes,
            "segment_count": len(segments),
            "gripper_transition_frames": {str(key): value for key, value in transitions.items()},
        }
        return ParsedSegments(segments=segments, source_manifest=manifest)

    @staticmethod
    def _predicates(action: str, obj: str, target: Optional[str], effector: str):
        if action == "pick":
            return (
                [Predicate("not_holding", [effector])],
                [Predicate("holding", [effector, obj])],
            )
        return (
            [Predicate("holding", [effector, obj])],
            [Predicate("inside", [obj, str(target)]), Predicate("holding", [effector, obj], False)],
        )
