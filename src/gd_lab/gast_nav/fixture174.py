"""Mixed easy-C primitive fixture: four tasks, same source solids for sensor/mesh."""
from dataclasses import replace
from types import SimpleNamespace
import math,numpy as np
from .inventory169 import PrimitiveCFixture169,fixture_meshes169,make_primitive_c_importer169
from .terrains import build_scene,KINDS
from .scene_types import StaticBox

def fixture174(kind,settings_path,seed=174):
    meshes,origin,meta=build_scene(kind,0.,SimpleNamespace(size=(24.,24.),settings_path=settings_path),seed=seed)
    solids=tuple(m.metadata['gast_nav_sensor_solid'] for m in meshes)
    # Explicit outer boundary; no unregistered clipped triangle border.
    border=tuple(StaticBox((x,y,.75),(hx,hy,.75),0.,'wall') for x,y,hx,hy in ((-11.94,0,.06,12),(11.94,0,.06,12),(0,-11.94,12,.06),(0,11.94,12,.06)))
    return PrimitiveCFixture169((24.,24.),solids+border,meta)

def configure_fixture174(terrain_cfg,n,settings_path,seed=174):
    from isaaclab.terrains import TerrainGenerator,TerrainGeneratorCfg
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
    from isaaclab.utils import configclass
    import trimesh
    class MixedGenerator(TerrainGenerator):
        def __init__(self,cfg,device='cpu'):
            self.counter174=0;self.scene_grid={};self.sensor_solids169={}
            super().__init__(cfg,device=device)
            cfg.nav_scene_grid=self.scene_grid;cfg.sensor_solids169=self.sensor_solids169;cfg.primitive_c169_sealed=True
        def _get_terrain_mesh(self,difficulty,cfg):
            index=self.counter174;self.counter174+=1
            fixture=fixtures[index]
            meshes,_=fixture_meshes169(fixture,positive_coordinates=False)
            mesh=trimesh.util.concatenate(meshes);mesh.metadata['fixture174']=fixture
            return mesh,np.zeros(3)
        def _add_sub_terrain(self,mesh,origin,row,col,sub_terrain_cfg):
            f=mesh.metadata['fixture174'];self.scene_grid[row,col]=f.meta;self.sensor_solids169[row,col]=f.solids
            return super()._add_sub_terrain(mesh,origin,row,col,sub_terrain_cfg)
        def _add_terrain_border(self):
            if self.cfg.border_width!=0:raise ValueError('implicit border forbidden')
    @configclass
    class Sub(SubTerrainBaseCfg):
        function=lambda difficulty,cfg:None  # generator overrides construction
    rows=math.ceil(math.sqrt(n));cols=math.ceil(n/rows)
    fixtures=[fixture174(KINDS[i%4],settings_path,seed+i) for i in range(rows*cols)]
    terrain_cfg.terrain_type='generator'
    terrain_cfg.terrain_generator=TerrainGeneratorCfg(class_type=MixedGenerator,seed=seed,size=(24.,24.),num_rows=rows,num_cols=cols,border_width=0.,curriculum=False,use_cache=False,color_scheme='none',sub_terrains={'mixed174':Sub(proportion=1.)})
    terrain_cfg.terrain_generator.expected_contact_boxes174=sum(len(f.solids) for f in fixtures)
    BaseImporter=make_primitive_c_importer169()
    class ContactGroundImporter174(BaseImporter):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            import omni.usd
            from pxr import UsdPhysics,UsdGeom,UsdShade,Gf
            stage=omni.usd.get_context().get_stage()
            # GPU filtered contact points do not support triangle-mesh targets.
            # Use the exact source boxes as physical colliders, retaining the
            # same triangle mesh only for terrain ray casts and rendering.
            mesh=stage.GetPrimAtPath('/World/ground/terrain/mesh')
            material,_=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial(materialPurpose='physics')
            if not material:raise RuntimeError('terrain physics material missing')
            UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Set(False)
            UsdGeom.Xform.Define(stage,'/World/ground/solids')
            origins=self.terrain_origins.detach().cpu().numpy().reshape(-1,3)
            gen=self.cfg.terrain_generator;count=0
            for tile,origin in enumerate(origins):
                for b in gen.sensor_solids169[(tile//gen.num_cols,tile%gen.num_cols)]:
                    cube=UsdGeom.Cube.Define(stage,f'/World/ground/solids/box_{count}');count+=1
                    cube.CreateSizeAttr(2.)
                    x=UsdGeom.Xformable(cube.GetPrim());x.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(b.center)+origin)))
                    x.AddRotateZOp().Set(math.degrees(b.yaw));x.AddScaleOp().Set(Gf.Vec3f(*b.half))
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose='physics')
                    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(True)
            gen.contact_boxes174=count
    terrain_cfg.class_type=ContactGroundImporter174
    return terrain_cfg
