#!/usr/bin/env python3
"""CI convention checks: name uniqueness, registration path, file size, comment policy."""

from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "gd_lab"
MAX_LINES = 300

# Shipped comments must not carry absolute paths or planning markers.
FORBIDDEN_COMMENT_PATTERNS = [
    r"/home/[a-z]",
    r"/Users/[A-Za-z]",
    r"\b(TODO|FIXME|XXX|HACK)\b",
    r"미구현|보류|추후|일단|임시",
]


def iter_py(base: pathlib.Path):
    yield from sorted(base.rglob("*.py"))


def check_unique_names() -> list[str]:
    errors = []
    seen: dict[str, pathlib.Path] = {}
    for path in iter_py(SRC):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                if name.startswith("_") or path.name == "__init__.py":
                    continue
                if name in seen and seen[name] != path:
                    errors.append(f"duplicate definition '{name}': {seen[name]} and {path}")
                else:
                    seen[name] = path
    return errors


def check_registration_path() -> list[str]:
    errors = []
    for path in iter_py(SRC):
        if path.name == "registry.py":
            continue
        text = path.read_text()
        if re.search(r"\bgym\.register\(", text):
            errors.append(f"direct gym.register outside core/registry.py: {path}")
    return errors


def check_file_size() -> list[str]:
    errors = []
    for path in iter_py(SRC):
        n = len(path.read_text().splitlines())
        if n > MAX_LINES:
            errors.append(f"{path} has {n} lines (cap {MAX_LINES})")
    return errors


def check_comment_policy() -> list[str]:
    errors = []
    for base in (SRC, ROOT / "scripts", ROOT / "tests"):
        for path in iter_py(base):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                for pattern in FORBIDDEN_COMMENT_PATTERNS:
                    if re.search(pattern, line, flags=re.IGNORECASE):
                        errors.append(f"{path}:{i}: forbidden reference ({pattern}): {line.strip()[:80]}")
    return errors


def check_no_vendored_rsl_rl() -> list[str]:
    hits = [p for p in ROOT.rglob("rsl_rl") if p.is_dir() and ".venv" not in p.parts]
    return [f"vendored rsl_rl copy found: {p}" for p in hits]


def main() -> int:
    errors = (
        check_unique_names()
        + check_registration_path()
        + check_file_size()
        + check_comment_policy()
        + check_no_vendored_rsl_rl()
    )
    for e in errors:
        print(f"ERROR: {e}")
    print(f"convention checks: {'FAILED' if errors else 'ok'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
