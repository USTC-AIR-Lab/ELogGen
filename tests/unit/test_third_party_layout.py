from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from eloggen.simulation.omnigibson import (
    behavior_root,
    data_root,
    project_root,
    robomimic_root,
    third_party_root,
)


def test_default_external_sources_live_under_third_party():
    with patch.dict(os.environ, {}, clear=True):
        root = project_root()
        assert data_root() == root
        assert third_party_root() == root / "third_party"
        assert behavior_root() == root / "third_party" / "BEHAVIOR-1K"
        assert robomimic_root() == root / "third_party" / "robomimic"


def test_data_root_moves_the_default_third_party_tree(tmp_path: Path):
    with patch.dict(
        os.environ,
        {"ELOGGEN_DATA_ROOT": str(tmp_path)},
        clear=True,
    ):
        assert data_root() == tmp_path.resolve()
        assert third_party_root() == (tmp_path / "third_party").resolve()
        assert behavior_root() == (tmp_path / "third_party" / "BEHAVIOR-1K").resolve()
        assert robomimic_root() == (tmp_path / "third_party" / "robomimic").resolve()


def test_explicit_source_roots_override_defaults(tmp_path: Path):
    third_party = tmp_path / "vendor"
    behavior = tmp_path / "behavior-custom"
    robomimic = tmp_path / "robomimic-custom"
    with patch.dict(
        os.environ,
        {
            "ELOGGEN_THIRD_PARTY_ROOT": str(third_party),
            "ELOGGEN_BEHAVIOR_ROOT": str(behavior),
            "ELOGGEN_ROBOMIMIC_ROOT": str(robomimic),
        },
        clear=True,
    ):
        assert third_party_root() == third_party.resolve()
        assert behavior_root() == behavior.resolve()
        assert robomimic_root() == robomimic.resolve()
