"""Environment-owned adapters; scene-owner modules remain unchanged."""
from collections import Counter
import torch
from .moving_cylinder import MovingCylinder, NoCylinderPathError
from .scene_types import SceneIndex


def generator_class():
    from .terrains import make_isaac_generator_class
    from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator
    class CombinedGenerator(make_isaac_generator_class(), GapMetadataTerrainGenerator):
        def __init__(self,cfg,device='cpu'):
            super().__init__(cfg,device=device)
            cfg.nav_scene_grid=self.scene_grid
    return CombinedGenerator


def bind_scene(terrain,index=None,env_ids=None):
    grid=terrain.cfg.terrain_generator.nav_scene_grid
    if index is None:
        return SceneIndex.from_grid(grid,terrain.terrain_levels,terrain.terrain_types)
    ids=list(range(len(index.env_tile_ids))) if env_ids is None else [int(i) for i in env_ids]
    mapping={key:i for i,key in enumerate(sorted(grid))}
    index.bind(ids,[mapping[(int(terrain.terrain_levels[i]),int(terrain.terrain_types[i]))] for i in ids])
    return index


class EpisodeCylinder(MovingCylinder):
    """Claude UPDATE160B: rejected path disables this episode, counted separately."""
    def __init__(self,*args,**kwargs):
        self.cylinder_skipped=Counter();self.cylinder_attempts=Counter()
        super().__init__(*args,**kwargs)

    def reset(self,env_ids=None,*,stage=None,robot_positions=None):
        if stage is not None:self.stage=stage
        ids=range(len(self.position)) if env_ids is None else [int(i) for i in env_ids]
        for i in ids:
            meta=self.index.lookup([i])[0];key=(meta.kind,meta.level)
            eligible=self.stage=='C' and meta.kind!='legacy'
            if eligible:self.cylinder_attempts[key]+=1
            try:super().reset([i],stage=self.stage,robot_positions=robot_positions)
            except NoCylinderPathError:
                self.cylinder_skipped[key]+=1
                self.enabled[i]=False;self.blocked[i]=False;self.initial_overlap[i]=False
                self.position[i]=self.env_origins[i]+self.position.new_tensor([0.,0.,self.settings['disabled_z_m']])
                self.endpoints[i]=self.position[i];self.velocity[i]=0.

    def update(self,dt,a_train_xy):
        """UPDATE161 B contract: called BEFORE step, in the fixed vehicle-floor xy frame.

        Until the scene-owner implementation arrives this is an explicit no-sway
        stub. It records the input, never silently invents a different sway model.
        """
        if dt<=0 or a_train_xy.shape!=(len(self.position),2) or not bool(torch.isfinite(a_train_xy).all()):
            raise ValueError('sway dt/acceleration')
        self.last_a_train_xy=a_train_xy.detach().clone()
        implementation=getattr(super(),'update',None)
        self.sway_api_stub=implementation is None
        if implementation is not None:return implementation(dt,a_train_xy)

    def skip_report(self):
        return [{'kind':k,'level':level,'attempts':n,'cylinder_skipped':self.cylinder_skipped[(k,level)],
                 'fraction':self.cylinder_skipped[(k,level)]/n,'over_10_percent':self.cylinder_skipped[(k,level)]/n>.1}
                for (k,level),n in sorted(self.cylinder_attempts.items())]
