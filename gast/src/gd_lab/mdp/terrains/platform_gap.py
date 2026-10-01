"""Two boarding gaps around a level, spawn-safe central platform."""

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from gd_lab.mdp.platform_gap_mesh import build_boarding_mesh


def platform_gap_terrain(difficulty: float, cfg: "PlatformGapTerrainCfg"):
    return build_boarding_mesh(
        difficulty, cfg.size, cfg.gap_width_range, cfg.height_offset_range,
        cfg.gap_center_offset, cfg.gap_depth,
    )


@configclass
class PlatformGapTerrainCfg(SubTerrainBaseCfg):
    function = platform_gap_terrain
    gap_width_range: tuple[float, float] = (0.02, 0.24)
    height_offset_range: tuple[float, float] = (0.0, 0.16)
    gap_center_offset: float = 1.5
    gap_depth: float = 0.65
