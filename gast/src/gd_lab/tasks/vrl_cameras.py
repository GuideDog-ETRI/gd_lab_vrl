"""Calibrated cameras for both teacher and student scene variants."""

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

from gd_lab.core.camera_contract import CAMERA_NAMES, load_camera_contract
from gd_lab.core.camera_timing import camera_period_steps


def default_vrl_camera(name: str) -> TiledCameraCfg:
    """Camera placeholder used during config construction."""
    contract = load_camera_contract()
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/vrl_" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=contract.focal_m * 1000,
            horizontal_aperture=contract.sensor_size_m[0] * 1000,
            clipping_range=contract.depth_clip,
        ),
        width=contract.width, height=contract.render_height,
        update_period=0.0, update_latest_camera_pose=True,
    )

def configure_vrl_cameras(cfg):
    """Call after Hydra overrides, before gym.make; respects the chosen profile."""
    contract = load_camera_contract(cfg.camera_profile)
    period_steps = camera_period_steps(
        cfg.decimation * cfg.sim.dt, contract.policy_dt, contract.period_steps
    )
    cfg.sim.render_interval = cfg.decimation * period_steps
    cfg.rerender_on_reset = True
    # CameraVisibleTerrain owns the global capture clock. A per-sensor period
    # would shift on reset and return stale data at the next global capture.
    # Zero means lazy data reads are fresh, NOT that we render each physics tick.
    for name, pos, quat in zip(CAMERA_NAMES, contract.positions, contract.quaternions_opengl, strict=True):
        norm = math.sqrt(sum(v*v for v in quat))
        setattr(cfg.scene, name, TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk/vrl_" + name,
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=tuple(v/norm for v in quat), convention="opengl"),
            data_types=["distance_to_image_plane", "rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=contract.focal_m * 1000,
                horizontal_aperture=contract.sensor_size_m[0] * 1000,
                clipping_range=contract.depth_clip,
            ),
            width=contract.width, height=contract.render_height,
            update_period=0.0,
            update_latest_camera_pose=True,
        ))
