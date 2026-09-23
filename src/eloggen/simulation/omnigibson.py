"""Activate the external simulator sources used by ElogGen."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def project_root() -> Path:
    configured = os.environ.get("ELOGGEN_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    configured = os.environ.get("ELOGGEN_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return project_root()


def third_party_root() -> Path:
    configured = os.environ.get("ELOGGEN_THIRD_PARTY_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (data_root() / "third_party").resolve()


def behavior_root() -> Path:
    configured = os.environ.get("ELOGGEN_BEHAVIOR_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (third_party_root() / "BEHAVIOR-1K").resolve()


def robomimic_root() -> Path:
    configured = os.environ.get("ELOGGEN_ROBOMIMIC_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (third_party_root() / "robomimic").resolve()


def activate() -> None:
    """Add the pinned BEHAVIOR-1K and robomimic checkouts to Python."""
    og_path = behavior_root() / "OmniGibson"
    if not og_path.is_dir():
        raise FileNotFoundError(f"OmniGibson directory not found: {og_path}")
    robomimic_path = robomimic_root()
    if not (robomimic_path / "robomimic").is_dir():
        raise FileNotFoundError(f"robomimic checkout not found: {robomimic_path}")

    for source_root in (robomimic_path, og_path):
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
