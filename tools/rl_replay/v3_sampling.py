"""24 B3 gap approaches, 4 native stairs with v2.1 pushes, 4 flat envs."""
import torch
from b3_sampling import ApproachSampler as B3Sampler,configure as configure_b3

def configure(cfg):
    from isaaclab.terrains import MeshPlaneTerrainCfg
    from gd_lab.mdp.gap_stair_v2 import set_gap_terrain_columns
    cfg.scene.terrain.terrain_generator.sub_terrains['v3_flat']=MeshPlaneTerrainCfg(proportion=1.)
    set_gap_terrain_columns(cfg,20)
    from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator
    import hashlib,json
    class V3Geometry(GapMetadataTerrainGenerator):
        def __init__(self,cfg,device='cpu'):
            super().__init__(cfg,device)
            cfg.v3_geometry_sha256=hashlib.sha256(self.terrain_mesh.vertices.tobytes()+self.terrain_mesh.faces.tobytes()).hexdigest()
            cfg.v3_gap_metadata_sha256=hashlib.sha256(json.dumps(cfg.gap_tile_metadata,sort_keys=True,default=str).encode()).hexdigest()
    cfg.scene.terrain.terrain_generator.class_type=V3Geometry
    configure_b3(cfg)

class ApproachSampler(B3Sampler):
    def __init__(self,env,gap):
        super().__init__(env,gap)
        from gd_lab.mdp.terrain_families import family_column_masks
        self.columns[28:32]=family_column_masks(env)['v3_flat'][0]
    def summary(self):
        return dict(super().summary(),distribution='24 gap + 4 native stairs (v2.1 handle force) + 4 independent flat')
