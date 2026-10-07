#!/usr/bin/env python3
"""Run CPU GAST unit tests without importing the IsaacLab task registry.

The selected Apptainer image supplies Python and PyTorch. Namespace-package
bootstrapping keeps these model/math tests independent of unrelated simulator
registration dependencies such as gymnasium and IsaacLab.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    source = repo / "src" / "gd_lab"
    tests = repo / "tests" / "gast" / "test_gast.py"
    _namespace("gd_lab", source)
    _namespace("gd_lab.core", source / "core")
    _namespace("gd_lab.gast", source / "gast")
    _namespace("gd_lab.students", source / "students")
    _namespace("gd_lab.students.gavd", source / "students" / "gavd")

    spec = importlib.util.spec_from_file_location("gast_test_suite", tests)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GAST tests from {tests}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.GastTests)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
