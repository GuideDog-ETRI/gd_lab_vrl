"""Replay-only runtime helpers (no Isaac or GPU initialization)."""
import fcntl
import math
import os
from pathlib import Path
import types


def validate_window(num_envs, seconds, warmup, vx):
    if type(num_envs) is not int or not 1 <= num_envs <= 128:
        raise ValueError("num_envs must be an integer in [1, 128]")
    for name, value, lo, hi in (("seconds", seconds, .01, 60), ("warmup", warmup, 0, 60)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{name} must be finite in [{lo}, {hi}]")
    if vx is not None and (isinstance(vx, bool) or not isinstance(vx, (int, float)) or not math.isfinite(vx) or not -2 <= vx <= 2):
        raise ValueError("vx must be finite in [-2, 2]")


def reserve_recording(out, live=None):
    """Shared CLI/server/worktree lock. Hold fd until process exit."""
    fd = os.open(f"/tmp/gd_lab_rl_replay_{os.getuid()}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for value in (out, live):
            if value and Path(value).exists():
                raise ValueError(f"refusing to overwrite replay output: {value}")
        if live and Path(out).resolve() == Path(live).resolve():
            raise ValueError("out and live must differ")
    except Exception:
        os.close(fd)
        raise
    return fd


def fix_velocity(term, vx):
    """Install before reset: enforce in command phase, never recompute history."""
    if vx is None:
        return
    def enforce():
        term.vel_command_b[:, 0] = vx
        term.vel_command_b[:, 1:] = 0
        for name in ("is_standing_env", "is_heading_env"):
            value = getattr(term, name, None)
            if value is not None:
                value[:] = False
    for name in ("_resample_command", "_update_command"):
        original = getattr(term, name)
        def wrapped(self, *args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            enforce()
            return result
        setattr(term, name, types.MethodType(wrapped, term))
    enforce()


def environment_metadata(base):
    actuators = getattr(base.scene["robot"], "actuators", {})
    def values(x):
        return x.detach().cpu().tolist() if hasattr(x, "detach") else x
    return {"TRAIN_ARM": os.environ.get("TRAIN_ARM"), "sim_dt": base.cfg.sim.dt,
            "decimation": base.cfg.decimation, "step_dt": base.step_dt,
            "actuator_gains": {name: {key: values(getattr(a, key, None))
                                     for key in ("stiffness", "damping")}
                               for name, a in actuators.items()}}


def terrain_capture_steps(base):
    """Read the actual term's clock; ordinary Ray PPO does not publish capture_steps."""
    configs = getattr(base.observation_manager, "_group_obs_term_cfgs", {}).get("terrain", [])
    candidates = []
    for cfg in configs:
        term = cfg.func
        if any(cls.__name__ in ("RaycastVisibleTerrainDropout", "CameraVisibleTerrain")
               for cls in type(term).__mro__):
            candidates.append(term.last_step)
    if len(candidates) == 1:
        return candidates[0]
    return None  # ambiguity is not a license to attach unrelated camera coordinates
