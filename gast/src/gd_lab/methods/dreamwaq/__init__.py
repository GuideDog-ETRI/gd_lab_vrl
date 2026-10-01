"""Blind DreamWaQ method: observation spec plus IsaacLab config fragments.

The config classes need IsaacLab and are loaded lazily, so the spec (and
everything derived from it) stays importable on machines without the simulator.
"""

from .spec import DREAMWAQ_SPEC, HISTORY_LENGTH, ONE_STEP_OBS_DIM, POLICY_OBS_DIM, VELOCITY_TARGET_SLICE

_LAZY = {
    "DreamwaqCurriculumCfg": "curriculum",
    "DreamwaqEventsCfg": "events",
    "DreamwaqObservationsCfg": "observations",
    "DreamwaqRewardsCfg": "rewards",
}

__all__ = [
    "DREAMWAQ_SPEC",
    "HISTORY_LENGTH",
    "ONE_STEP_OBS_DIM",
    "POLICY_OBS_DIM",
    "VELOCITY_TARGET_SLICE",
    *_LAZY.keys(),
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
