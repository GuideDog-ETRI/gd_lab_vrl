"""Four static scene builders, plus lazy Isaac TerrainGenerator adapters.

Callbacks match Isaac (difficulty,cfg)->(list[Trimesh],origin). Builders also
return SceneTileMeta. Meshes are in [0,size] before Isaac centres them; metadata
is always relative to origin == tile centre z=0. No Isaac import at module load.
"""
from dataclasses import asdict
import math
import numpy as np
import trimesh
from .scene_config import load_scene_config
from .scene_types import StaticBox, SceneTileMeta
from .scene_geometry import point_box_distance, width_profile

KINDS = ('nav_corridor','nav_door','nav_turn_room','nav_stair_landing')
META_KEY = 'gast_nav_scene'


def _uniform(rng, bounds): return float(rng.uniform(*bounds))
def _narrow(bounds, difficulty): return float(bounds[1]+difficulty*(bounds[0]-bounds[1]))
def _count(rng, bounds): return int(rng.integers(bounds[0], bounds[1]+1))


def build_scene(kind, difficulty, cfg=None, *, seed=None):
    """Return (positive-coordinate meshes, centre origin, env-local metadata).

    cfg: Isaac subterrain config or any object with optional size, seed,
    settings_path. Explicit seed is deterministic and does not change global RNG.
    """
    if kind not in KINDS or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError('scene kind/difficulty')
    settings = load_scene_config(getattr(cfg,'settings_path',None))
    t = settings['terrain']; size = tuple(getattr(cfg,'size',t['size']))
    if len(size) != 2 or min(size) <= 0: raise ValueError('tile size')
    seed = seed if seed is not None else getattr(cfg,'seed',None)
    rng = np.random.default_rng(seed)
    level = min(t['levels']-1, int(difficulty*t['levels']))
    obstacles, solids = [], []
    def box(center, half, yaw=0., kind='box', *, obstacle=True):
        value = StaticBox(tuple(float(x) for x in center),tuple(float(x) for x in half),float(yaw),kind)
        solids.append(value)
        if obstacle: obstacles.append(value)
        return value
    def floor_slab(a,b,top=0.):
        if b-a <= t['geometry_epsilon']: return
        box(((a+b)/2,0.,top-t['floor_thickness']/2),((b-a)/2,size[1]/2,t['floor_thickness']/2),kind='step',obstacle=False)
    wall_h = _uniform(rng,t['wall_height']); thick = t['wall_thickness']
    landing, turn_spot = None, None
    stair_floor = lambda xy: 0.
    if kind != 'nav_door': floor_slab(-size[0]/2,size[0]/2)
    if kind == 'nav_corridor':
        c = settings['corridor']; width = _narrow(c['width'],difficulty)
        length = _uniform(rng,c['length']); angle = math.radians(_uniform(rng,c['bend_deg']))
        route = ((-length/2,0.),(0.,0.),(length/2*math.cos(angle),length/2*math.sin(angle)))
        route_length = length
        for a,b in zip(route,route[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]; segment=math.hypot(dx,dy); yaw=math.atan2(dy,dx)
            nx,ny=-dy/segment,dx/segment
            for side in (-1,1):
                box(((a[0]+b[0])/2+side*nx*(width+thick)/2,(a[1]+b[1])/2+side*ny*(width+thick)/2,wall_h/2),
                    (segment/2,thick/2,wall_h/2),yaw,'wall')
        for _ in range(_count(rng,c['box_count'])):
            s = _uniform(rng,(0.,length)); leg=0 if s <= length/2 else 1
            a,b=route[leg],route[leg+1]; yaw=math.atan2(b[1]-a[1],b[0]-a[0]); along=s-leg*length/2
            depth=_uniform(rng,c['box_depth']); extent=_uniform(rng,c['box_length']); h=_uniform(rng,c['box_height']); side=rng.choice([-1,1])
            x=a[0]+along*math.cos(yaw)-side*math.sin(yaw)*(width-depth)/2
            y=a[1]+along*math.sin(yaw)+side*math.cos(yaw)*(width-depth)/2
            box((x,y,h/2),(extent/2,depth/2,h/2),yaw,'box')
        if rng.random() < c['pillar_probability']:
            p=_uniform(rng,c['pillar_width']); x=route[0][0]+length/4
            box((x,-(width-p)/2,wall_h/2),(p/2,p/2,wall_h/2),kind='pillar')
        approach=c['approach_length']
        a,b=route[-2:]; yaw=math.atan2(b[1]-a[1],b[0]-a[0])
        route=((route[0][0]-approach,route[0][1]),)+route+((b[0]+approach*math.cos(yaw),b[1]+approach*math.sin(yaw)),)
        segments=(('PASSAGE',0.,route_length+2*approach),)
        nominal=lambda s: width if approach-t['geometry_epsilon'] <= s <= approach+length+t['geometry_epsilon'] else max(width,c['approach_width'])
    elif kind == 'nav_door':
        c=settings['door']; width=_narrow(c['width'],difficulty); depth=_uniform(rng,c['thickness']); h=_uniform(rng,c['height'])
        gap=_uniform(rng,c['gap']); offset=c['gap_offset']; threshold=_uniform(rng,c['threshold'])
        if not offset > gap/2+depth/2: raise ValueError('gap overlaps doorway')
        edges=[(-size[0]/2,-offset-gap/2),(-offset+gap/2,offset-gap/2),(offset+gap/2,size[0]/2)]
        for a,b in edges: floor_slab(a,b)
        if gap > t['geometry_epsilon']:
            for x in (-offset,offset): floor_slab(x-gap/2,x+gap/2,-c['gap_depth'])
        for side in (-1,1):
            span=(size[1]-width)/2
            box((0.,side*(width/2+span/2),h/2),(depth/2,span/2,h/2),kind='frame')
        box((0.,0.,h+c['header_height']/2),(depth/2,width/2,c['header_height']/2),kind='frame')
        if threshold > t['geometry_epsilon']: box((0.,0.,threshold/2),(depth/2,width/2,threshold/2),kind='step')
        length=c['route_length']; route=((-length/2,0.),(length/2,0.)); segments=(('PASSAGE',0.,length),); nominal=size[1]
    elif kind == 'nav_turn_room':
        from .turn_clutter import build_turn_clutter
        clutter,route,nominal,radius=build_turn_clutter(difficulty,level,rng,settings,wall_h)
        for primitive in clutter:
            box(primitive.center,primitive.half,primitive.yaw,primitive.kind)
        turn_spot=(route[-1],radius)
        route_length=math.dist(route[0],route[-1])
        segments=(('TURN_SPOT',0.,route_length),)

    else:
        c=settings['stairs']; width=_narrow(c['width'],difficulty); rise=c['rise'][0]+difficulty*(c['rise'][1]-c['rise'][0])
        run=_uniform(rng,c['run']); steps=_count(rng,c['steps']); flat=_narrow(c['landing_length'],difficulty); top=steps*rise
        up=-flat/2-steps*run; down=flat/2
        stair_steps=[]
        for i in range(steps):
            z=(i+1)*rise
            stair_steps.append(box((up+(i+.5)*run,0.,z/2),(run/2,width/2,z/2),kind='step'))
        stair_steps.append(box((0.,0.,top/2),(flat/2,width/2,top/2),kind='step'))
        for i in range(steps):
            z=(steps-i)*rise
            stair_steps.append(box((down+(i+.5)*run,0.,z/2),(run/2,width/2,z/2),kind='step'))
        def stair_floor(xy):
            return max([0.]+[b.center[2]+b.half[2] for b in stair_steps if abs(xy[0]-b.center[0]) <= b.half[0]+t['geometry_epsilon'] and abs(xy[1]) <= width/2])
        route_start=up-c['approach_length']; route_end=down+steps*run+c['approach_length']; length=route_end-route_start
        route=((route_start,0.),(route_end,0.)); landing=((-flat/2,-width/2),(flat/2,width/2))
        for side in (-1,1):
            box(((route_start+route_end)/2,side*(width+thick)/2,(top+wall_h)/2),(length/2,thick/2,(top+wall_h)/2),kind='wall')
            # Shared StaticBox is yaw-only: use stepwise horizontal rails with
            # vertical posts, not an unrepresented pitched collision primitive.
            for step in stair_steps:
                x=step.center[0]; z=step.center[2]+step.half[2]; rt=c['rail_thickness']
                box((x,side*(width/2-rt/2),z+c['rail_height']),(step.half[0],rt/2,rt/2),kind='rail')
                box((x,side*(width/2-rt/2),z+c['rail_height']/2),(rt/2,rt/2,c['rail_height']/2),kind='rail')
        for _ in range(_count(rng,c['box_count'])):
            i=int(rng.integers(0,steps)); step=stair_steps[i]
            depth=_uniform(rng,c['box_depth']); extent=min(run,_uniform(rng,c['box_length'])); h=_uniform(rng,c['box_height']); side=rng.choice([-1,1])
            box((step.center[0],side*(width-depth)/2,step.center[2]+step.half[2]+h/2),(extent/2,depth/2,h/2),kind='box')
        before=c['approach_length']; landing_start=before+steps*run; landing_end=landing_start+flat
        segments=(('FLAT',0.,before),('STAIR_UP',before,landing_start),('LANDING',landing_start,landing_end),('STAIR_DOWN',landing_end,landing_end+steps*run),('FLAT',landing_end+steps*run,length)); nominal=width
    # A too-small requested tile fails rather than silently cropping geometry.
    for b in solids:
        cosine,sine=abs(math.cos(b.yaw)),abs(math.sin(b.yaw))
        ex,ey=cosine*b.half[0]+sine*b.half[1],sine*b.half[0]+cosine*b.half[1]
        if abs(b.center[0])+ex > size[0]/2+t['geometry_epsilon'] or abs(b.center[1])+ey > size[1]/2+t['geometry_epsilon']:
            raise ValueError('scene does not fit cfg.size; enlarge tile')
    profile=width_profile(route,obstacles,nominal,floor_fn=stair_floor,height=settings['oracle']['collision_height_m'],epsilon=t['profile_epsilon'])
    meta=SceneTileMeta(kind,level,tuple(route),profile,tuple(segments),landing,turn_spot,tuple(obstacles),'stairs' if kind=='nav_stair_landing' else 'flat',kind in t['inertia_kinds'])
    origin=np.array([size[0]/2,size[1]/2,0.],dtype=float); meshes=[]
    for b in solids:
        transform=trimesh.transformations.rotation_matrix(b.yaw,[0,0,1])
        transform[:3,3]=np.array(b.center)+origin
        # trimesh.apply_transform may probe winding with global NumPy RNG on
        # rotations. This is a proper yaw rotation: transform vertices directly
        # so terrain sampling never perturbs another family's RNG stream.
        mesh=trimesh.creation.box(extents=np.array(b.half)*2)
        mesh.vertices=np.asarray(mesh.vertices)@transform[:3,:3].T+transform[:3,3]
        mesh.metadata['gast_nav_sensor_solid']=b
        meshes.append(mesh)
    meshes[0].metadata[META_KEY]=asdict(meta)
    return meshes,origin,meta


def nav_corridor_terrain(difficulty,cfg): return build_scene('nav_corridor',difficulty,cfg)[:2]
def nav_door_terrain(difficulty,cfg): return build_scene('nav_door',difficulty,cfg)[:2]
def nav_turn_room_terrain(difficulty,cfg): return build_scene('nav_turn_room',difficulty,cfg)[:2]
def nav_stair_landing_terrain(difficulty,cfg): return build_scene('nav_stair_landing',difficulty,cfg)[:2]
FUNCTIONS=dict(zip(KINDS,(nav_corridor_terrain,nav_door_terrain,nav_turn_room_terrain,nav_stair_landing_terrain)))


def make_isaac_cfg(kind, *, settings_path=None, size=None, proportion=1.):
    """Lazy real SubTerrainBaseCfg; call only after the Isaac app starts."""
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
    from isaaclab.utils import configclass
    if kind not in FUNCTIONS: raise ValueError('kind')
    @configclass
    class GastNavTerrainCfg(SubTerrainBaseCfg):
        function=FUNCTIONS[kind]
        settings_path: str | None=None
    cfg=GastNavTerrainCfg()
    cfg.settings_path=settings_path; cfg.size=tuple(size or load_scene_config(settings_path)['terrain']['size']); cfg.proportion=proportion
    return cfg


def make_isaac_generator_class():
    """TerrainGenerator subclass retaining metadata at actual (row,col).

    Use use_cache=False. Plain Isaac concatenation/OBJ cache loses arbitrary
    metadata; this hook explicitly preserves it. Legacy generation is untouched.
    """
    from isaaclab.terrains import TerrainGenerator
    class GastNavTerrainGenerator(TerrainGenerator):
        def __init__(self,*args,**kwargs):
            self.scene_grid={}; self.sensor_grid={}; self._nav_serial=0
            super().__init__(*args,**kwargs)
        def _get_terrain_mesh(self,difficulty,cfg):
            matches=[k for k,f in FUNCTIONS.items() if cfg.function is f]
            if not matches: return super()._get_terrain_mesh(difficulty,cfg)
            if self.cfg.use_cache: raise ValueError('GAST-Nav metadata requires use_cache=False')
            local=cfg.copy(); base=self.cfg.seed
            local.seed=None if base is None else int(base)+self._nav_serial
            self._nav_serial+=1
            meshes,origin,meta=build_scene(matches[0],difficulty,local)
            mesh=trimesh.util.concatenate(meshes)
            translation=np.array([-local.size[0]/2,-local.size[1]/2,0.])
            mesh.apply_translation(translation); origin=origin+translation
            mesh.metadata[META_KEY]=meta
            mesh.metadata['gast_nav_sensor_solids']=tuple(m.metadata['gast_nav_sensor_solid'] for m in meshes)
            return mesh,origin
        def _add_sub_terrain(self,mesh,origin,row,col,sub_terrain_cfg):
            meta=mesh.metadata.get(META_KEY)
            if meta is not None:
                self.scene_grid[(row,col)]=meta
                self.sensor_grid[(row,col)]=mesh.metadata['gast_nav_sensor_solids']
            else:
                # LEGACY metadata is a tag/binding only, not a replacement for
                # its existing terrain-specific oracle or command generator.
                sx,sy=self.cfg.size
                self.scene_grid[(row,col)]=SceneTileMeta('legacy',row,((-sx/4,0.),(sx/4,0.)),((0.,sy),(sx/2,sy)),(('FLAT',0.,sx/2),),None,None,(),'flat',False)
            return super()._add_sub_terrain(mesh,origin,row,col,sub_terrain_cfg)
    return GastNavTerrainGenerator
