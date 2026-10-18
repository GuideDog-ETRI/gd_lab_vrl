"""UPDATE160 CPU-only shared metadata stub. Production must use scene_types/terrains.
This module does not synthesize pretend Isaac terrain or replace the scene owner's files.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class StaticBox:
    center: tuple
    half: tuple
    yaw: float
    kind: str

@dataclass(frozen=True)
class SceneTileMeta:
    kind: str
    level: int
    route_polyline: tuple
    width_profile: tuple
    segments: tuple
    landing: tuple | None
    turn_spot: tuple | None
    obstacles: tuple
    floor_z_at: str
    train_inertia: bool

class SceneIndex:
    def __init__(self, tiles): self.tiles = list(tiles)
    def lookup(self, env_ids): return [self.tiles[int(i)] for i in env_ids]
    def tensors(self, device):
        import torch, math
        return {'route_axis_yaw':torch.tensor([math.atan2(t.route_polyline[1][1]-t.route_polyline[0][1],
                    t.route_polyline[1][0]-t.route_polyline[0][0]) for t in self.tiles], device=device),
                'width':torch.tensor([t.width_profile[0][1] for t in self.tiles],device=device),
                'landing_mask':torch.tensor([t.landing is not None for t in self.tiles],device=device)}
