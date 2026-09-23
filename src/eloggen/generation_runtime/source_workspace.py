"""Filesystem workspace helpers for source-dataset replay.

The robomimic / OmniGibson preprocessing path can mutate the HDF5 passed to it.
Task-pack ``source.hdf5`` files are raw inputs and must remain immutable, so all
replay/preprocessing happens on either the requested processed output or a
temporary copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


@dataclass
class ReplayDatasetWorkspace:
    """A replay-safe HDF5 path and its optional temporary directory."""

    source_path: Path
    replay_path: Path
    persistent: bool
    _temporary_directory: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> Path:
        return self.replay_path

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def create_replay_dataset_workspace(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    persistent: bool = False,
) -> ReplayDatasetWorkspace:
    """Create a copy that may safely be preprocessed or replayed.

    ``persistent=True`` is used by ``source prepare``: the source is copied to
    ``processed.hdf5`` (or an explicit output path), and that copy becomes the
    preprocessing / replay target.

    ``persistent=False`` is used by ``source inspect-boundaries``: a hidden
    temporary copy is created next to the source so any in-place preprocessing
    cannot alter the task-pack raw source. The copy is removed by ``cleanup`` or
    by the context-manager exit path.
    """

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source HDF5 not found: {source}")

    if persistent:
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source.with_name("processed.hdf5")
        )
        if destination == source:
            raise ValueError("source and processed output paths must be different")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return ReplayDatasetWorkspace(
            source_path=source,
            replay_path=destination,
            persistent=True,
        )

    temporary_directory = tempfile.TemporaryDirectory(
        prefix=".eloggen_replay_",
        dir=str(source.parent),
    )
    replay_path = Path(temporary_directory.name) / source.name
    shutil.copy2(source, replay_path)
    return ReplayDatasetWorkspace(
        source_path=source,
        replay_path=replay_path,
        persistent=False,
        _temporary_directory=temporary_directory,
    )
