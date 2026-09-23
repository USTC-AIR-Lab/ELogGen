from pathlib import Path
import tempfile

import pytest

from eloggen.generation_runtime.source_workspace import create_replay_dataset_workspace


def test_inspect_workspace_mutates_only_temporary_copy():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.hdf5"
        source.write_bytes(b"raw-source")

        workspace = create_replay_dataset_workspace(source, persistent=False)
        replay_path = workspace.replay_path
        replay_parent = replay_path.parent

        assert replay_path != source
        assert replay_path.read_bytes() == b"raw-source"

        # Stand in for robomimic / OmniGibson in-place preprocessing.
        replay_path.write_bytes(b"preprocessed-copy")
        assert source.read_bytes() == b"raw-source"

        workspace.cleanup()
        assert not replay_parent.exists()
        assert source.read_bytes() == b"raw-source"


def test_prepare_workspace_writes_processed_copy_not_source():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.hdf5"
        processed = root / "processed.hdf5"
        source.write_bytes(b"raw-source")

        workspace = create_replay_dataset_workspace(
            source,
            output_path=processed,
            persistent=True,
        )

        assert workspace.replay_path == processed.resolve()
        assert processed.read_bytes() == b"raw-source"

        # Stand in for preprocessing and datagen-info writes.
        processed.write_bytes(b"processed-output")
        assert source.read_bytes() == b"raw-source"

        workspace.cleanup()
        assert processed.read_bytes() == b"processed-output"


def test_processed_output_cannot_alias_raw_source():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.hdf5"
        source.write_bytes(b"raw-source")

        with pytest.raises(ValueError, match="must be different"):
            create_replay_dataset_workspace(
                source,
                output_path=source,
                persistent=True,
            )
