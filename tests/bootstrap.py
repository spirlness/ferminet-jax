"""Backward-compatible bootstrap module for legacy test imports."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import project_paths  # type: ignore  # noqa: F401

project_paths.ensure_project_paths()
