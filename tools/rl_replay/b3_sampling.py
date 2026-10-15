"""B3 sampling only; privileged geometry never enters the student or actor."""
import types
import torch

def visible_ground_extent(root_height,profile='vendor_new'):
    """Conservative corner-ray extent on a level deck at the upright reset pose."""
    from gd_lab.core.camera_contract import load_camera_contract
    c=load_camera_contract(profile)
    px=torch.tensor([0.,80.,0.,80.]);py=torch.tensor([0.,0.,45.,45.])
    ray=torch.stack(((px-40)/c.fx,-(py-22.5)/c.fy,-torch.ones(4)),-1)
    extent=0.
    for quat,pos in zip(c.quaternions_opengl,c.positions):
        q=torch.tensor(quat,dtype=torch.float32);q=q/q.norm();v=q[1:].expand_as(ray)
        r=ray+2*(q[0]*torch.cross(v,ray,dim=-1)+torch.cross(v,torch.cross(v,ray,dim=-1),dim=-1))
        t=-(root_height+pos[2])/r[:,2]
        # Infinite ray extent is conservative relative to depth range clipping.
        valid=(r[:,2]<0)&(t>0)
        if valid.any():extent=max(extent,float((pos[0]+t*r[:,0])[valid].max()))
    return extent

def approach_done(env):
    sampler=getattr(env,'_b3_sampler',None)
    return sampler.done() if sampler is not None else torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)

def configure(cfg):
    from isaaclab.managers import TerminationTermCfg
    cfg.terminations.b3_approach=TerminationTermCfg(func=approach_done,time_out=True)

class ApproachSampler:
    def __init__(self,env,gap):
        from gd_lab.mdp.terrain_families import family_column_masks
        self.env=env;self.gap=gap
        if env.num_envs!=32:raise ValueError('B3 requires 32 environments')
        self.visible_extent=visible_ground_extent(float(env.scene['robot'].data.default_root_state[:24,2].max()))
        if self.visible_extent+.15>=2.:raise ValueError('Camera sees too far for the configured pre-visible spawn; revisit sampling geometry')
        cols=family_column_masks(env);gap_cols=cols['platform_gap']
        stairs=torch.cat([v for k,v in cols.items() if 'stairs' in k])
        others=torch.cat([v for k,v in cols.items() if k!='platform_gap' and 'stairs' not in k])
        if not len(gap_cols) or not len(stairs) or not len(others):raise ValueError('B3 needs gap, stairs and other terrain columns')
        self.columns=torch.cat((gap_cols[torch.arange(24,device=env.device)%len(gap_cols)],
            stairs[torch.arange(4,device=env.device)%len(stairs)],others[torch.arange(4,device=env.device)%len(others)]))
        self.best=torch.zeros(32,device=env.device);self.last_progress=torch.zeros_like(self.best)
        self.reset_counts=torch.zeros(32,dtype=torch.long,device=env.device)
        self.passed=self.stalled=0
        self.original=env._reset_idx
        def reset(base,ids,*args,**kwargs):
            self.assign(ids)
            result=self.original(ids,*args,**kwargs)
            self.place(ids)
            return result
        env._reset_idx=types.MethodType(reset,env);env._b3_sampler=self
    def assign(self,ids):
        b=self.env;t=b.scene.terrain
        t.terrain_types[ids]=self.columns[ids]
        t.env_origins[ids]=t.terrain_origins[t.terrain_levels[ids],t.terrain_types[ids]]
        b.scene.env_origins[ids]=t.env_origins[ids]
    def place(self,ids):
        b=self.env;ids=ids[ids<24];robot=b.scene['robot']
        if not len(ids):return
        slots,_=self.gap.for_envs(b)
        # Start on the middle deck >2m before the leading edge, outside the
        # student's forward .8m reconstruction grid; native camera FOV is unchanged.
        distance=2.+.2*torch.rand(len(ids),device=b.device)
        x=slots[ids,1,0]-distance
        if not (x>slots[ids,0,1]+.35).all():raise ValueError('No safe pre-approach spawn on middle deck')
        pose=robot.data.default_root_state[ids,:7].clone()
        pose[:,:3]+=b.scene.env_origins[ids];pose[:,0]=x+b.scene.env_origins[ids,0]
        pose[:,1]=b.scene.env_origins[ids,1];pose[:,3:]=0;pose[:,3]=1
        robot.write_root_pose_to_sim(pose,env_ids=ids)
        robot.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=b.device),env_ids=ids)
        self.best[ids]=pose[:,0];self.last_progress[ids]=b.common_step_counter*b.step_dt
        self.reset_counts[ids]+=1
    def done(self):
        b=self.env;x=b.scene['robot'].data.root_pos_w[:,0];now=b.common_step_counter*b.step_dt
        progress=x>self.best+.05
        self.best=torch.where(progress,x,self.best);self.last_progress=torch.where(progress,now,self.last_progress)
        slots,_=self.gap.for_envs(b);passed=x>slots[:,1,1]+b.scene.env_origins[:,0]+.8
        stalled=now-self.last_progress>=2.
        active=torch.arange(32,device=b.device)<24
        self.passed+=int((passed&active).sum());self.stalled+=int((stalled&active&~passed).sum())
        return active&(passed|stalled)
    def summary(self):return {'columns':self.columns.cpu().tolist(),'gap_reset_counts':self.reset_counts[:24].cpu().tolist(),'passed_resets':self.passed,'stall_resets':self.stalled,'nominal_visible_ground_extent_m':self.visible_extent,'spawn_distance_m':[2.,2.2]}
    def close(self):self.env._reset_idx=self.original;del self.env._b3_sampler
