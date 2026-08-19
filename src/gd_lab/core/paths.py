"""Filesystem anchors. This module imports nothing from gd_lab."""

from __future__ import annotations

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

ASSETS_DIR = os.path.join(_REPO_ROOT, "assets")

LOG_ROOT = os.environ.get("GD_LAB_LOG_ROOT", os.path.join(_REPO_ROOT, "logs"))
