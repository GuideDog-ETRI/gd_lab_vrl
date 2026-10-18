"""Stage-C single kinematic cylinder; synchronized physical/observed state.

No simulator imports. The map capsule is a conservative bounding proxy for the
flat-ended physical cylinder: segment endpoints at physical bottom/top, same
radius. It encloses the cylinder and extends by radius at each end (documented).
"""
import math
import numpy as np
import torch
from .map_types import DynamicCapsules
from .scene_config import load_scene_config
from .scene_geometry import point_box_distance, floor_height, line_box_interval
from .scene_types import route_point


class NoCylinderPathError(ValueError):
    """Explicit reset rejection; owning env records before approved resampling."""
    def __init__(self,env_id,meta,radius,attempts):
        self.env_id=env_id; self.kind=meta.kind; self.level=meta.level
        self.radius_m=radius; self.attempts=attempts
        super().__init__(f'NO_CYLINDER_PATH env={env_id} kind={meta.kind} level={meta.level} radius={radius:.4f} attempts={attempts}')


class MovingCylinder:
    def __init__(self,index,*,stage='C',seed=0,device='cpu',env_origins=None,config=None):
        self.index=index; self.config=config or load_scene_config(); self.settings=self.config['cylinder']
        self.device=torch.device(device); self.rng=np.random.default_rng(seed)
        n=len(index.env_tile_ids)
        self.env_origins=torch.zeros(n,3,device=device) if env_origins is None else torch.as_tensor(env_origins,device=device,dtype=torch.float32).clone()
        if self.env_origins.shape!=(n,3): raise ValueError('env origin shape')
        self.position=torch.zeros(n,3,device=device); self.velocity=torch.zeros_like(self.position)
        self.endpoints=torch.zeros(n,2,3,device=device); self.radius=torch.zeros(n,device=device)
        self.height=torch.full((n,),self.settings['height'],device=device); self.speed=torch.zeros(n,device=device)
        self.target=torch.ones(n,dtype=torch.long,device=device); self.pingpong=torch.zeros(n,dtype=torch.bool,device=device)
        self.enabled=torch.zeros(n,dtype=torch.bool,device=device); self.blocked=torch.zeros_like(self.enabled)
        self.initial_overlap=torch.zeros_like(self.enabled)
        self.stage=stage; self.reset()

    def reset(self,env_ids=None,*,stage=None,robot_positions=None):
        if stage is not None: self.stage=stage
        if self.stage not in ('A0','A','B','C'): raise ValueError('stage')
        ids=list(range(len(self.position))) if env_ids is None else [int(i) for i in env_ids]
        metas=self.index.lookup(ids); cfg=self.settings
        if robot_positions is None:
            all_meta=self.index.lookup(range(len(self.position)))
            robot_positions=torch.tensor([[*(m.turn_spot[0] if m.turn_spot else m.route_polyline[0]),0.] for m in all_meta],device=self.device)+self.env_origins
        for i,meta in zip(ids,metas):
            self.radius[i]=float(self.rng.uniform(*cfg['radius'])); self.speed[i]=float(self.rng.uniform(*cfg['speed']))
            self.pingpong[i]=self.rng.random()<cfg['pingpong_probability']; self.target[i]=1
            self.blocked[i]=False; self.initial_overlap[i]=False
            self.enabled[i]=self.stage=='C' and meta.kind!='legacy'
            if not bool(self.enabled[i]):
                self.position[i]=self.env_origins[i]+torch.tensor([0.,0.,cfg['disabled_z_m']],device=self.device)
                self.endpoints[i]=self.position[i]; continue
            found=False
            for _ in range(cfg['sample_attempts']):
                # A local straight segment; stairs keep the cylinder on the
                # broad landing plane rather than teleporting it over risers.
                if meta.kind=='nav_turn_room':
                    clearance=meta.turn_spot[1]-float(self.radius[i])-cfg['clearance_m']
                    if clearance<=cfg['endpoint_epsilon_m']: break
                    theta=self.rng.uniform(0.,2*math.pi,2)
                    radial=clearance*np.sqrt(self.rng.uniform(0.,1.,2))
                    a=(meta.turn_spot[0][0]+float(radial[0]*math.cos(theta[0])),meta.turn_spot[0][1]+float(radial[0]*math.sin(theta[0])))
                    b=(meta.turn_spot[0][0]+float(radial[1]*math.cos(theta[1])),meta.turn_spot[0][1]+float(radial[1]*math.sin(theta[1])))
                elif meta.landing is not None:
                    r=float(self.radius[i])+cfg['clearance_m']
                    surfaces=[]
                    (x0,y0),(x1,y1)=meta.landing
                    if min(x1-x0,y1-y0)>2*r:
                        y=float(self.rng.uniform(y0+r,y1-r)); surfaces.append(((x0+r,y),(x1-r,y)))
                    for name,lo,hi in meta.segments:
                        if name=='FLAT' and hi-lo>2*r:
                            pa,_=route_point(meta,lo+r); pb,_=route_point(meta,hi-r); surfaces.append((pa,pb))
                    if not surfaces: break
                    a,b=surfaces[int(self.rng.integers(0,len(surfaces)))]
                else:
                    lo,hi=sorted(self.rng.uniform(0.,meta.route_length,2))
                    a,_=route_point(meta,lo); b,_=route_point(meta,hi)
                z=floor_height(meta,a)
                radius=float(self.radius[i])+cfg['clearance_m']
                # Endpoints and complete swept segment must avoid static OBBs.
                # Box Minkowski inflation is conservative at rounded corners.
                safe=True
                for obstacle in meta.obstacles:
                    if obstacle.kind=='step' or obstacle.center[2]+obstacle.half[2] <= z: continue
                    delta=(b[0]-a[0],b[1]-a[1]); length=math.hypot(*delta)
                    if point_box_distance(a,obstacle) <= radius or point_box_distance(b,obstacle) <= radius:
                        safe=False; break
                    if length>cfg['endpoint_epsilon_m']:
                        from .scene_types import StaticBox
                        inflated=StaticBox(obstacle.center,(obstacle.half[0]+radius,obstacle.half[1]+radius,obstacle.half[2]),obstacle.yaw,obstacle.kind)
                        interval=line_box_interval(a,(delta[0]/length,delta[1]/length),inflated,cfg['endpoint_epsilon_m'])
                        if interval and interval[1]>=0 and interval[0]<=length: safe=False; break
                if not safe: continue
                start=torch.tensor([a[0],a[1],z+cfg['height']/2],device=self.device)+self.env_origins[i]
                if robot_positions is not None:
                    distance=torch.linalg.vector_norm(start[:2]-robot_positions[i,:2])
                    if float(distance) <= radius+cfg['robot_radius_m']: continue
                self.endpoints[i,0]=start
                self.endpoints[i,1]=torch.tensor([b[0],b[1],z+cfg['height']/2],device=self.device)+self.env_origins[i]
                self.position[i]=start; found=True; break
            if not found:
                # No safe spawn/path is explicit, not a silent overlap. Caller
                # must resample the scene/mark invalid; C requires one cylinder.
                raise NoCylinderPathError(i,meta,float(self.radius[i]),cfg['sample_attempts'])
        self.velocity[ids]=0.

    def step(self,dt,robot_positions,robot_velocities=None):
        """Continuous relative-motion collision check, not endpoint-only crossing.

        Whole step stops if its swept centre segment intersects the robot's
        expanded footprint. Existing overlap is flagged, not fixed by teleport.
        Robot motion that independently strikes a stopped collider remains an
        episode contact/failure to be counted by the environment.
        """
        if not math.isfinite(dt) or dt<=0: raise ValueError('dt')
        if robot_positions.shape!=self.position.shape: raise ValueError('robot positions')
        if robot_velocities is None: robot_velocities=torch.zeros_like(robot_positions)
        if robot_velocities.shape!=self.position.shape: raise ValueError('robot velocities')
        goal=self.endpoints[torch.arange(len(self.position),device=self.device),self.target]
        delta=goal-self.position; distance=torch.linalg.vector_norm(delta,dim=-1)
        travel=torch.minimum(self.speed*dt,distance)
        movement=delta/distance.clamp_min(self.settings['endpoint_epsilon_m'])[:,None]*travel[:,None]
        relative=self.position[:,:2]-robot_positions[:,:2]
        relative_step=movement[:,:2]-robot_velocities[:,:2]*dt
        sq=relative_step.square().sum(-1)
        u=(-(relative*relative_step).sum(-1)/sq.clamp_min(torch.finfo(sq.dtype).eps)).clamp(0,1)
        closest=torch.linalg.vector_norm(relative+u[:,None]*relative_step,dim=-1)
        limit=self.radius+self.settings['robot_radius_m']+self.settings['clearance_m']
        vertical=((self.position[:,2]-self.height/2 <= robot_positions[:,2]+self.config['oracle']['collision_height_m'])
                  & (self.position[:,2]+self.height/2 >= robot_positions[:,2]))
        self.initial_overlap=(torch.linalg.vector_norm(relative,dim=-1)<=limit)&vertical&self.enabled
        self.blocked=(closest<=limit)&vertical&self.enabled
        move=torch.where((self.enabled&~self.blocked)[:,None],movement,0.)
        self.position+=move; self.velocity=move/dt
        reached=(torch.linalg.vector_norm(goal-self.position,dim=-1)<=self.settings['endpoint_epsilon_m'])&self.enabled&~self.blocked
        self.target=torch.where(reached&self.pingpong,1-self.target,self.target)
        return self.position.clone(),self.blocked.clone()

    def capsules(self):
        """DynamicCapsules[N,1,...] in WORLD coordinates; transform to sample frame before maps."""
        if not bool(self.enabled.any()): return None
        a=self.position.clone(); b=self.position.clone()
        a[:,2]-=self.height/2; b[:,2]+=self.height/2
        return DynamicCapsules(a[:,None],b[:,None],self.radius[:,None].clone())

    def collider_specs(self):
        """Physical cylinder dimensions; create/scale each env's collider accordingly on reset."""
        return [{'radius':float(r),'height':float(h),'kinematic_enabled':True,'enabled':bool(e)}
                for r,h,e in zip(self.radius,self.height,self.enabled)]

    def write_colliders(self,rigid_object,env_ids=None):
        """Duck-typed Isaac RigidObject adapter; root pose uses (x,y,z,w,x,y,z).

        Object must have KINEMATIC cylinder geometry matching collider_specs().
        Radius randomization requires the owning env to rebuild/scale shapes on
        reset before this pose/velocity write (not silently left at old radius).
        """
        ids=torch.arange(len(self.position),device=self.device) if env_ids is None else torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        pose=torch.zeros(len(ids),7,device=self.device); pose[:,:3]=self.position[ids]; pose[:,3]=1.
        rigid_object.write_root_pose_to_sim(pose,env_ids=ids)
        velocity=torch.zeros(len(ids),6,device=self.device); velocity[:,:3]=self.velocity[ids]
        rigid_object.write_root_velocity_to_sim(velocity,env_ids=ids)
