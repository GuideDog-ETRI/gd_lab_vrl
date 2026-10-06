"""Attach exact gap bounds and deck heights to the shared terrain cfg while the tiles are generated."""

from isaaclab.terrains.terrain_generator import TerrainGenerator

from gd_lab.mdp.platform_gap_metadata import boarding_tile_metadata
from gd_lab.mdp.terrains.platform_gap import PlatformGapTerrainCfg


class GapMetadataTerrainGenerator(TerrainGenerator):
    """Attach exact per-tile geometry to the shared config at insertion time."""

    def __init__(self, cfg, device="cpu"):
        cfg.gap_tile_metadata = [[None for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]
        super().__init__(cfg, device=device)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if isinstance(sub_terrain_cfg, PlatformGapTerrainCfg):
            metadata = boarding_tile_metadata(mesh, sub_terrain_cfg.gap_center_offset, self.cfg.seed)
            metadata["row"] = row
            metadata["col"] = col
            self.cfg.gap_tile_metadata[row][col] = metadata
        return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
