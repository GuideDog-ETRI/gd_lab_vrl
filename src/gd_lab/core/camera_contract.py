"""Versioned camera calibration shared by observation, noise and export code.

Transforms are camera-to-trunk, wxyz quaternions in OpenGL convention.
Numbers are transcribed from the vendor MuJoCo BT0..BT3 bodies, not guessed
from camera link names. The two vendor releases are deliberately distinct.
"""

from dataclasses import asdict, dataclass
from math import atan, degrees

CAMERA_NAMES = ("front_depth_camera0", "front_depth_camera1", "hind_depth_camera2", "hind_depth_camera3")
DEFAULT_CAMERA_PROFILE = "vendor_legacy"
TERRAIN_OBSERVATION_VERSION = 2


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
