"""Checkpoint package metadata lookup helpers shared by student launchers."""

from pathlib import Path


def find_teacher_env_yaml(checkpoint_path):
    """Find env metadata beside a checkpoint or in its staged teacher package."""
    checkpoint = Path(checkpoint_path).resolve()
    for parent in checkpoint.parents:
        for package_dir in ("params", "teacher_params"):
            candidate = parent / package_dir / "env.yaml"
            if candidate.is_file():
                return candidate
    return None
