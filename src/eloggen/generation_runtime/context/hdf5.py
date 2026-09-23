"""HDF5 read / write helpers for frame context annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np

from eloggen.generation_runtime.context.frames import frame_contexts_to_json


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


def write_frame_context_json(ep_data_grp: h5py.Group, frame_contexts: Iterable[Dict[str, Any]]) -> None:
    frame_contexts = list(frame_contexts)
    num_samples = int(ep_data_grp.attrs.get("num_samples", len(frame_contexts)))
    if len(frame_contexts) != num_samples:
        raise ValueError(f"frame_context_json length {len(frame_contexts)} != num_samples {num_samples}")

    context_grp = ep_data_grp.require_group("context")
    if "frame_context_json" in context_grp:
        del context_grp["frame_context_json"]
    dt = h5py.string_dtype(encoding="utf-8")
    context_grp.create_dataset(
        "frame_context_json",
        data=np.asarray(frame_contexts_to_json(frame_contexts), dtype=dt),
    )


def read_frame_context_json(hdf5_path: str, demo_key: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as dataset:
        return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[f"data/{demo_key}/context/frame_context_json"][:]]


def has_frame_context(hdf5_path: str, demo_key: str) -> bool:
    with h5py.File(hdf5_path, "r") as dataset:
        return f"data/{demo_key}/context/frame_context_json" in dataset


def write_frame_context_report(
    output_path: str,
    frame_contexts: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any] | None = None,
) -> str:
    frame_contexts = list(frame_contexts)
    stage_counts: Dict[str, int] = {}
    for context in frame_contexts:
        stage_type = str(context.get("stage_type", "unknown"))
        stage_counts[stage_type] = stage_counts.get(stage_type, 0) + 1

    payload = {
        "metadata": metadata or {},
        "summary": {
            "num_frames": len(frame_contexts),
            "stage_type_counts": stage_counts,
        },
        "frame_contexts": frame_contexts,
    }
    payload = _as_jsonable(payload)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")
    return str(path)
