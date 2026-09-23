"""Public collection contract without a bundled hardware implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


COLLECTOR_ENTRYPOINT_GROUP = "eloggen.collectors"


@dataclass(frozen=True)
class CollectionRequest:
    task_name: str
    output: Path
    options: dict[str, Any]


@dataclass(frozen=True)
class CollectionResult:
    source_hdf5: Path
    episodes: int
    metadata: dict[str, Any]


@runtime_checkable
class Collector(Protocol):
    """Interface implemented by optional teleoperation plugins."""

    name: str

    def collect(self, request: CollectionRequest) -> CollectionResult:
        ...


def available_collectors() -> dict[str, Any]:
    collectors: dict[str, Any] = {}
    discovered = entry_points()
    selected = discovered.select(group=COLLECTOR_ENTRYPOINT_GROUP)
    for item in selected:
        collectors[item.name] = item
    return collectors


def load_collector(name: str) -> Collector:
    collectors = available_collectors()
    if name not in collectors:
        available = ", ".join(sorted(collectors)) or "none"
        raise LookupError(
            f"Collection backend {name!r} is not installed. "
            f"Available plugin backends: {available}. Existing source HDF5 files "
            "can be used without a teleoperation backend."
        )
    collector = collectors[name].load()()
    if not isinstance(collector, Collector):
        raise TypeError(f"Plugin {name!r} does not implement the Collector protocol")
    return collector
