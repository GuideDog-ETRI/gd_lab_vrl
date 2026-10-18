"""C2 runtime thresholds, loaded without simulator imports."""
from dataclasses import dataclass
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class MapConfig:
    ttl_s: float
    ray_max_range_m: float
    ray_chunk: int
    cell_chunk: int
    intersection_epsilon: float
    max_step_m: float
    max_slope_deg: float
    horizontal_max_slope_deg: float
    min_z_m: float
    max_z_m: float
    flat_support_tolerance_m: float
    overlap_epsilon: float


def load_config(path=None):
    """Load configs/gast_nav/maps.yaml or an explicit deployment path."""
    path = Path(path) if path else Path(__file__).resolve().parents[3] / 'configs/gast_nav/maps.yaml'
    data = yaml.safe_load(path.read_text())
    cfg = MapConfig(**data['map'], **data['support'], **data['warp'])
    for name, value in vars(cfg).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f'nonfinite config: {name}')
    if not (cfg.ttl_s > 0 and cfg.ray_max_range_m > 0 and cfg.ray_chunk > 0 and cfg.cell_chunk > 0
            and cfg.intersection_epsilon > 0 and cfg.overlap_epsilon > 0
            and cfg.min_z_m < cfg.max_z_m and cfg.max_step_m > 0 and cfg.flat_support_tolerance_m > 0
            and 0 <= cfg.max_slope_deg < 90 and 0 <= cfg.horizontal_max_slope_deg < 90):
        raise ValueError('invalid map thresholds')
    return cfg
