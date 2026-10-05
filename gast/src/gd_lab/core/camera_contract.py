"""Versioned camera calibration shared by observation, noise and export code.

Transforms are camera-to-trunk, wxyz quaternions in OpenGL convention (the camera
looks along -z, +y up). Numbers are transcribed from the vendor MuJoCo BT0..BT3
bodies (``rbq_simulator/rbq_mujoco/resources/model/rbq/rbq.xml``), not guessed from
camera link names. The two vendor releases are deliberately distinct:

* ``vendor_new`` (default): the RBQ SDK values -- identical in v1.19.47, v1.20.0,
  public ``main`` and the nightly builds of 2026-09-09 (41f6fac6) and 2026-10-04
  (5974087c). All four ground-view cameras look downward; cam0-2 rotations equal
  the official manual's base->camera TF, cam3 differs from it by about 11.7 deg.
* ``vendor_legacy``: only the 2026-08-29 nightly (814e6d4d). Positions equal the
  manual, but every rotation differs from the manual and from all other SDK
  releases by 90 deg (cameras look up/level). Keep it ONLY to load or evaluate
  checkpoints trained with it; starting new training with it needs an explicit
  opt-in (``require_training_camera_profile``).
"""

import os
from dataclasses import asdict, dataclass, replace
from math import atan, degrees

CAMERA_NAMES = ("front_depth_camera0", "front_depth_camera1", "hind_depth_camera2", "hind_depth_camera3")
DEFAULT_CAMERA_PROFILE = "vendor_new"
LEGACY_CAMERA_PROFILES = ("vendor_legacy",)
ALLOW_LEGACY_CAMERA_ENV = "GD_LAB_ALLOW_LEGACY_CAMERA"  # start training with vendor_legacy
ALLOW_MISSING_CAMERA_CONTRACT_ENV = "GD_LAB_ALLOW_MISSING_CAMERA_CONTRACT"  # accept a checkpoint without one
PROFILE_SOURCES = {
    "vendor_new": ("RBQ SDK rbq.xml BT0-3 (v1.19.47, v1.20.0, main, nightly 41f6fac6 2026-09-09, nightly 5974087c "
                   "2026-10-04); main commit 68bc33b77719d357b4323fb88549efd905caf721, rbq.xml git blob "
                   "68ad7e0ac2d5bf181d3c6b3663310ff582894f8c"),
    "vendor_legacy": "RBQ nightly 814e6d4d 2026-08-29 rbq.xml BT0-3 only; rotations differ by 90 deg from the manual",
}
TERRAIN_OBSERVATION_VERSION = 2


def camera_contract_for_policy(profile: str, policy_dt: float):
    """Record the actual control rate while preserving the physical camera interval."""
    from gd_lab.core.camera_timing import camera_period_steps

    contract = load_camera_contract(profile)
    steps = camera_period_steps(policy_dt, contract.policy_dt, contract.period_steps)
    return replace(contract, policy_dt=policy_dt, period_steps=steps)


@dataclass(frozen=True)
class VrlCameraContract:
    profile: str
    positions: tuple
    quaternions_opengl: tuple
    sensor_size_m: tuple
    focal_m: float = 0.00193
    width: int = 80
    height: int = 45
    source_resolution: tuple = (640, 360)
    depth_clip: tuple = (0.15, 5.0)
    period_steps: int = 4
    policy_dt: float = 0.02
    # Isaac's pinhole renderer assumes fx == fy. Overscan then resample to
    # the vendor's fx/fy instead of trusting its ignored vertical aperture.
    render_height: int = 48

    @property
    def fx(self):
        return self.width * self.focal_m / self.sensor_size_m[0]

    @property
    def fy(self):
        return self.height * self.focal_m / self.sensor_size_m[1]

    @property
    def fov_degrees(self):
        return tuple(degrees(2 * atan(s / (2 * self.focal_m))) for s in self.sensor_size_m)

    def manifest(self):
        return {
            **asdict(self), "schema_version": 1, "terrain_observation_version": TERRAIN_OBSERVATION_VERSION,
            "camera_names": list(CAMERA_NAMES), "channels": ["depth", "ir_proxy"],
            "transform": "camera_to_trunk", "camera_convention": "opengl: looks along -z, +y up",
            "source": PROFILE_SOURCES.get(self.profile),
            "depth_semantics": "optical_axis_metres", "quaternion_order": "wxyz",
            "fov_degrees": self.fov_degrees, "fx": self.fx, "fy": self.fy,
            "cx": self.width / 2, "cy": self.height / 2,
            "frame_shape": [1, 4, 2, self.height, self.width],
        }


def load_camera_contract(profile: str = DEFAULT_CAMERA_PROFILE) -> VrlCameraContract:
    if profile == "vendor_legacy":
        return VrlCameraContract(
            profile,
            ((0.36462, 0, -0.02663), (0.26053, 0, -0.04759),
             (-0.19515, 0.0065, -0.04832), (-0.352990, -0.000011, -0.020510)),
            ((0, 0.8191608, 0, -0.5735639), (0, -0.6156417, 0, 0.7880262),
             (0, -0.7071046, 0, 0.7071090), (0.4993997, 0.0263259, 0.8647709, -0.0455865)),
            (0.003663, 0.0021396),
        )
    if profile == "vendor_new":
        return VrlCameraContract(
            profile,
            ((0.364, 0, -0.024919), (0.26097, 0, -0.04582),
             (-0.19515, 0.0065, -0.0465), (-0.352082, -0.000011, -0.018938)),
            ((0, -0.1736482, 0, 0.9848078), (0, 0.1218693, 0, 0.9925462),
             (0, 0, 0, 1), (0.9848078, 0, 0.1736482, 0)),
            (0.003896, 0.002140),
        )
    raise ValueError(f"Unknown camera profile {profile!r}; use vendor_legacy or vendor_new")


_IDENTITY_FIELDS = ("profile", "positions", "quaternions_opengl", "sensor_size_m", "focal_m", "width", "height",
                    "source_resolution", "depth_clip", "camera_names", "depth_semantics")
_TIMING_FIELDS = ("policy_dt", "period_steps")


def _freeze(value):
    return tuple(_freeze(v) for v in value) if isinstance(value, (list, tuple)) else value


def camera_identity(contract, timing=True) -> tuple:
    """Calibration (+ camera order, depth semantics, optionally timing) of a contract or saved manifest.

    Descriptive metadata (``source``, ``transform`` ...) is ignored on purpose, so adding such fields
    never invalidates an existing checkpoint whose geometry is unchanged.
    """
    manifest = contract if isinstance(contract, dict) else contract.manifest()
    keys = _IDENTITY_FIELDS + (_TIMING_FIELDS if timing else ())
    return tuple(_freeze(manifest.get(key)) for key in keys)


def same_camera_contract(a, b, timing=True) -> bool:
    return camera_identity(a, timing) == camera_identity(b, timing)


def check_checkpoint_camera_contract(recorded, expected=None, *, purpose="use", environ=None):
    """Fail closed on a checkpoint's recorded camera contract (resume, evaluation, export).

    A checkpoint WITHOUT a contract cannot prove which calibration it learned: refused unless
    ``GD_LAB_ALLOW_MISSING_CAMERA_CONTRACT=1`` explicitly accepts such an unverified file (a separate
    decision from ``GD_LAB_ALLOW_LEGACY_CAMERA``, which only lets legacy training start). With ``expected``,
    the recorded contract must have the same identity (calibration, camera order, depth semantics, timing).
    """
    environ = os.environ if environ is None else environ
    if recorded is None:
        if environ.get(ALLOW_MISSING_CAMERA_CONTRACT_ENV) != "1":
            raise ValueError(f"checkpoint has no camera contract, so its calibration cannot be verified for {purpose}; "
                             f"set {ALLOW_MISSING_CAMERA_CONTRACT_ENV}=1 only to accept an unverified checkpoint")
        return None
    if expected is not None and not same_camera_contract(recorded, expected):
        raise ValueError(f"checkpoint camera contract differs from the one required for {purpose}")
    return recorded


def require_training_camera_profile(profile: str, environ=None) -> None:
    """Refuse to START training/distillation with a legacy calibration unless explicitly allowed.

    Loading and evaluating legacy checkpoints stays possible; only new optimisation is gated.
    """
    environ = os.environ if environ is None else environ
    if profile in LEGACY_CAMERA_PROFILES and environ.get(ALLOW_LEGACY_CAMERA_ENV) != "1":
        raise ValueError(
            f"camera_profile={profile!r} is the 2026-08-29 calibration whose rotations disagree with the RBQ SDK "
            f"and manual; train with 'vendor_new', or set {ALLOW_LEGACY_CAMERA_ENV}=1 to reproduce an old run"
        )
