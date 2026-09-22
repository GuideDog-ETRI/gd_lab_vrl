"""Process-local Kit storage; never change the shared image or other runs."""

import os
from pathlib import Path


def prepare_vrl_runtime(args):
    """Keep Kit caches off Apptainer's small writable-tmpfs overlay.

    AppLauncher splits kit_args on whitespace; reject unsupported paths rather
    than silently redirecting only part of a path. Each process has its own
    directory so concurrent training never shares writable cache files.
    """
    root = Path(__file__).resolve().parents[1] / "logs" / "kit_runtime" / str(os.getpid())
    if any(c.isspace() for c in str(root)):
        raise ValueError("VRL runtime directory must not contain whitespace")
    flags = []
    for token in ("cache", "data", "logs"):
        directory = root / token
        directory.mkdir(parents=True, exist_ok=True)
        flags.append(f"--/app/tokens/{token}={directory}")
    mpl = root / "matplotlib"
    mpl.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    args.kit_args = " ".join([getattr(args, "kit_args", "") or "", *flags]).strip()
    print(f"[INFO] Process-local Kit runtime: {root}")
    return root


def verify_vrl_runtime(root):
    """Check the effective cache token after Kit starts, before training."""
    import carb.tokens

    actual = Path(carb.tokens.get_tokens_interface().resolve("${cache}"))
    if not actual.is_relative_to(root):
        raise RuntimeError(f"Kit cache override was not applied: {actual}")
    print(f"[INFO] Verified Kit cache: {actual}")
