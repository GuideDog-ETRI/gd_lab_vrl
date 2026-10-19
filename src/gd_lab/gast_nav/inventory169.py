"""Closed, primitive-only static C fixture. One source for meshes and sensors.

No legacy terrain, no automatic clipped border mesh, no implicit ground plane.
The factory explicitly builds floor and four border cuboids. Every collision
triangle is from exactly one registered cuboid. This is a different diagnostic
fixture, not an approximate representation of the old mixed terrain.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .scene_types import StaticBox,SceneTileMeta
from .producer168 import SensorPrimitives

KEY='primitive_c169'
@dataclass(frozen=True)
class PrimitiveCFixture169:
    size: tuple
    solids: tuple
    meta: SceneTileMeta


def primitive_c_fixture169(size=(24.,24.)):
    if len(size)!=2 or min(size)<24.:raise ValueError('fixture tiles must be at least 24m for range-isolated sensors')
    sx,sy=size;t=.12;h=1.5
    solids=[StaticBox((0.,0.,-.1),(sx/2,sy/2,.1),0.,'step')]
    for side in (-1,1):
        solids.append(StaticBox((0.,side*1.36,h/2),(5.,t/2,h/2),0.,'wall'))
        solids.append(StaticBox((side*(sx/2-t/2),0.,h/2),(t/2,sy/2,h/2),0.,'wall'))
        solids.append(StaticBox((0.,side*(sy/2-t/2),h/2),(sx/2,t/2,h/2),0.,'wall'))
    # Minimum-diameter fixture obstacle off the centre route, not hidden from inventory.
    solids.append(StaticBox((1.,.95,.35),(.025,.025,.35),0.,'pillar'))
    obstacles=tuple(solids[1:])
    meta=SceneTileMeta('nav_corridor',0,((-2.,0.),(2.,0.)),((0.,2.6),(4.,2.6)),(('PASSAGE',0.,4.),),None,None,obstacles,'flat',False)
    return PrimitiveCFixture169(tuple(size),tuple(solids),meta)


def fixture_meshes169(fixture,*,positive_coordinates=True):
    import trimesh
    shift=np.array([fixture.size[0]/2,fixture.size[1]/2,0.]) if positive_coordinates else np.zeros(3)
    meshes=[]
    for index,box in enumerate(fixture.solids):
        c,s=math.cos(box.yaw),math.sin(box.yaw);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        mesh=trimesh.creation.box(extents=np.array(box.half)*2)
        mesh.vertices=np.asarray(mesh.vertices)@r.T+np.array(box.center)+shift
        mesh.metadata[KEY]={'solid_id':index,'center':box.center,'half':box.half,'yaw':box.yaw}
        meshes.append(mesh)
    return meshes,shift


def make_primitive_c_generator169():
    """Lazy Isaac generator. Do not mix this class with a legacy generator.

    scene_grid/nav_scene_grid preserve the existing scene binding contract.
    sensor_solids169 records ALL local solids including floor and explicit border.
    Each environment must bind a UNIQUE tile; duplicate tile assignment raises
    in sensor inventory. This diagnostic uses no legacy curriculum.
    """
    import trimesh
    from isaaclab.terrains import TerrainGenerator
    class PrimitiveCGenerator169(TerrainGenerator):
        def __init__(self,cfg,device='cpu'):
            if cfg.use_cache or cfg.border_width!=0:raise ValueError('C169 requires use_cache=False,border_width=0')
            if any(v.function is not primitive_c_terrain169 for v in cfg.sub_terrains.values()):raise ValueError('C169 supports ONLY primitive fixture subterrain')
            self.scene_grid={};self.sensor_solids169={}
            super().__init__(cfg,device=device)
            cfg.nav_scene_grid=self.scene_grid;cfg.sensor_solids169=self.sensor_solids169
            cfg.primitive_c169_sealed=True
        def _get_terrain_mesh(self,difficulty,cfg):
            fixture=primitive_c_fixture169(tuple(cfg.size));meshes,_=fixture_meshes169(fixture,positive_coordinates=False)
            mesh=trimesh.util.concatenate(meshes);mesh.metadata[KEY]=fixture
            return mesh,np.zeros(3)
        def _add_sub_terrain(self,mesh,origin,row,col,sub_terrain_cfg):
            fixture=mesh.metadata[KEY];self.scene_grid[row,col]=fixture.meta;self.sensor_solids169[row,col]=fixture.solids
            return super()._add_sub_terrain(mesh,origin,row,col,sub_terrain_cfg)
        def _add_terrain_border(self):
            # Border cuboids are explicitly included in every fixture. Isaac's
            # clipped triangle border would violate the box inventory contract.
            if self.cfg.border_width!=0:raise ValueError('implicit border forbidden')
    return PrimitiveCGenerator169


def primitive_c_terrain169(difficulty,cfg):
    fixture=primitive_c_fixture169(tuple(cfg.size));meshes,origin=fixture_meshes169(fixture)
    return meshes,origin


def primitive_c_generator_cfg169(num_envs,*,seed=169):
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
    from isaaclab.utils import configclass
    if num_envs<1:raise ValueError('num_envs')
    @configclass
    class SubCfg169(SubTerrainBaseCfg):
        function=primitive_c_terrain169
    rows=math.ceil(math.sqrt(num_envs));cols=math.ceil(num_envs/rows)
    return TerrainGeneratorCfg(class_type=make_primitive_c_generator169(),seed=seed,size=(24.,24.),num_rows=rows,num_cols=cols,border_width=0.,curriculum=False,use_cache=False,color_scheme='none',sub_terrains={'primitive_c169':SubCfg169(proportion=1.)})


def primitive_c_inventory169(fixture,env_origins,camera_origins_w,k,*,max_m,
                             all_body_bounds_xy_w,unique_tile_ids,sealed=False,
                             no_external_colliders=False,image_size=(80,45)):
    """Return WORLD SensorPrimitives, before producer168_adapter adds own self.

    all_body_bounds_xy_w [N,2,2] is the actual WHOLE collision AABB min/max in
    world xy for every robot (not just link centres). Guard foreign robots in
    each tile's inner +/-4m area. If any robot violates this, ALL envs invalid.
    Capture queries must stay strictly inside their tile (sphere encloses every
    optical frustum); foreign floors/borders and robots then cannot be first hit.
    sealed/no_external_colliders are asserted by the actual scene installer,
    never inferred just from a scene label. Defaults fail closed.
    """
    n=len(env_origins);device=env_origins.device;dtype=env_origins.dtype
    if env_origins.shape!=(n,3) or camera_origins_w.ndim!=3 or camera_origins_w.shape[0]!=n or camera_origins_w.shape[2]!=3:raise ValueError('origin shapes')
    if all_body_bounds_xy_w.shape!=(n,2,2):raise ValueError('whole-body collision xy bounds required')
    if unique_tile_ids.shape!=(n,) or unique_tile_ids.dtype!=torch.int64 or unique_tile_ids.device!=device:raise ValueError('tile IDs')
    if len(torch.unique(unique_tile_ids))!=n:raise ValueError('C169 forbids duplicate/shared tile assignment')
    if not (0<max_m<math.inf):raise ValueError('range')
    # Tile origins must lie on this exact nonoverlapping 24m lattice. A compact
    # neighbour/legacy layout cannot silently opt into the isolation guarantee.
    d=env_origins[:,None,:2]-env_origins[None,:,:2]
    sx,sy=fixture.size
    lattice=(d[:,:,0]/sx-(d[:,:,0]/sx).round()).abs()<1e-5
    lattice &= (d[:,:,1]/sy-(d[:,:,1]/sy).round()).abs()<1e-5
    distinct=(d.abs().amax(-1)>1e-4)|torch.eye(n,device=device,dtype=torch.bool)
    layout_ok=bool((lattice&distinct).all())
    if k.ndim==3:k=k[None].expand(n,-1,-1,-1)
    if k.shape[:2]!=camera_origins_w.shape[:2] or k.shape[-2:]!=(3,3):raise ValueError('K shape')
    if not bool(torch.isfinite(k).all()&(k[:,:,0,0]>0).all()&(k[:,:,1,1]>0).all()):raise ValueError('K finite/focal')
    width,height=image_size
    if width<1 or height<1:raise ValueError('image size')
    u=torch.stack(((-.5-k[:,:,0,2])/k[:,:,0,0],(width-.5-k[:,:,0,2])/k[:,:,0,0]),-1).abs().amax(-1)
    v=torch.stack(((-.5-k[:,:,1,2])/k[:,:,1,1],(height-.5-k[:,:,1,2])/k[:,:,1,1]),-1).abs().amax(-1)
    reach=max_m*torch.sqrt(1+u*u+v*v)
    camera_xy=(camera_origins_w[:,:,:2]-env_origins[:,None,:2]).abs()
    query_fits=(camera_xy+reach[:,:,None]<env_origins.new_tensor([sx/2,sy/2])-1e-4).all((1,2))
    body_local=all_body_bounds_xy_w-env_origins[:,None,:2]
    body_safe=bool((torch.isfinite(body_local)&(body_local.abs()<=4.)).all())
    nprim=len(fixture.solids)
    centres=env_origins[:,None]+env_origins.new_tensor([b.center for b in fixture.solids])[None]
    half=env_origins.new_tensor([b.half for b in fixture.solids])[None].expand(n,-1,-1).clone()
    angles=env_origins.new_tensor([b.yaw for b in fixture.solids]);c,s=angles.cos(),angles.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
    rotation=torch.stack((c,-s,z,s,c,z,z,z,one),-1).reshape(1,nprim,3,3).expand(n,-1,-1,-1).clone()
    complete=query_fits & (sealed and no_external_colliders and body_safe and layout_ok)
    return SensorPrimitives(centres,rotation,half,torch.zeros((n,nprim),device=device,dtype=torch.long),torch.ones((n,nprim),device=device,dtype=torch.bool),complete)


def bind_unique_c_tiles169(terrain,num_envs):
    """Call BEFORE the initial robot reset; disable terrain curriculum thereafter.

    Isaac's default origin sampler can put multiple robots in one tile. This
    fixture requires one tile/env so that foreign self geometry is range-isolated.
    InteractiveScene.env_origins reads terrain.env_origins directly.
    """
    if not getattr(terrain.cfg.terrain_generator,'primitive_c169_sealed',False):raise ValueError('not a C169 generator')
    origins=terrain.terrain_origins
    rows,cols=origins.shape[:2]
    if num_envs>rows*cols:raise ValueError('insufficient unique fixture tiles')
    ids=torch.arange(num_envs,device=origins.device);levels=ids//cols;types=ids%cols
    terrain.terrain_levels=levels;terrain.terrain_types=types
    terrain.env_origins[:num_envs]=origins[levels,types]
    return ids


def self_collision_bounds_xy169(body_link_pos_w,body_link_quat_w,body_names,specs):
    """Whole actual self-collider OBB enclosure, not link-centre bounds."""
    from gd_lab.core.camera_geometry import camera_rotation_matrix
    ids=torch.tensor([body_names.index(s.link) for s in specs],device=body_link_pos_w.device)
    r=camera_rotation_matrix(body_link_quat_w[:,ids])
    centres=body_link_pos_w[:,ids]+torch.einsum('npij,pj->npi',r,body_link_pos_w.new_tensor([s.xyz for s in specs]))
    rotations=r@body_link_pos_w.new_tensor([s.rotation for s in specs])
    extent=(rotations.abs()*body_link_pos_w.new_tensor([s.half for s in specs])[None,:,None,:]).sum(-1)
    return torch.stack(((centres[...,:2]-extent[...,:2]).amin(1),(centres[...,:2]+extent[...,:2]).amax(1)),1)


def make_primitive_c_importer169():
    """Bind unique origins while constructing the scene, before any robot reset."""
    from isaaclab.terrains import TerrainImporter
    class PrimitiveCImporter169(TerrainImporter):
        def configure_env_origins(self,origins=None):
            if origins is None:raise ValueError('C169 requires its exact generated primitive mesh')
            super().configure_env_origins(origins)
            bind_unique_c_tiles169(self,self.cfg.num_envs)
        def update_terrain_levels(self,*args,**kwargs):
            raise ValueError('C169 unique-tile diagnostic forbids terrain curriculum reassignment')
    return PrimitiveCImporter169


def configure_primitive_c_terrain169(terrain_cfg,num_envs,*,seed=169):
    """Apply LAST, after legacy/GAST terrain setup, before constructing the env.

    Caller must disable terrain-level curriculum and additional static/dynamic
    collider spawns. This function does not assert no_external_colliders=True.
    """
    terrain_cfg.terrain_type='generator'
    terrain_cfg.terrain_generator=primitive_c_generator_cfg169(num_envs,seed=seed)
    terrain_cfg.class_type=make_primitive_c_importer169()
    return terrain_cfg
