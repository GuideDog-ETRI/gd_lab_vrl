"""Closed-loop E1 adapter. Imported only by explicit --e1_condition replay."""
import hashlib,json,types,os
from pathlib import Path
import torch
from e1_core import LatentSwitch,action_at_boundary
from e1_metrics import EpisodeMetrics
from runtime import fix_velocity,environment_metadata
from e1_v3_adapter import E1V3Driver
from gd_lab.gast.observations import pose_xyyaw

def load_teacher(env,cfg,path,device,policy_class=None):
    # Construct the network only: no PPO runner/storage/optimizer, no CENet Adam.
    import importlib
    from rsl_rl.utils import resolve_obs_groups
    config=dict(cfg['policy']);name=config.pop('class_name')
    if policy_class is None:
        module,attr=name.split(':');policy_class=getattr(importlib.import_module(module),attr)
    obs=env.get_observations()
    groups=resolve_obs_groups(obs,cfg['obs_groups'],['critic'])
    policy=policy_class(obs,groups,env.num_actions,inference_only=True,**config).to(device)
    saved=torch.load(path,map_location=device,weights_only=False)
    policy.load_state_dict(saved['model_state_dict'],strict=True)
    if policy.cenet.optimizer is not None:raise RuntimeError('E1 created a CENet optimizer')
    policy.eval()
    for p in policy.parameters():p.requires_grad_(False)
    return policy

def configure(env_cfg):
    # Student v21 changes widths/commands but, unlike the teacher task, does not
    # install this metadata hook. Set it BEFORE gym.make generates the mesh.
    # It delegates mesh generation unchanged and records exact insertion geometry.
    from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator
    generator=env_cfg.scene.terrain.terrain_generator
    if 'platform_gap' not in generator.sub_terrains:
        raise ValueError('E1 requires platform_gap terrain')
    generator.class_type=GapMetadataTerrainGenerator
    # Keep the gap-focus distribution (20 gap columns), but fix terrain levels
    # across conditions; intervention outcomes must not drive curriculum changes.
    if hasattr(env_cfg.curriculum,'terrain_levels'):env_cfg.curriculum.terrain_levels=None
    env_cfg.scene.terrain.max_init_terrain_level=env_cfg.scene.terrain.terrain_generator.num_rows-1

class Recorder:
    def __init__(self,base,switch,condition,geometry,family):
        self.base=base;self.switch=switch;self.geometry=geometry;self.family=family
        self.robot=base.scene['robot'];self.sensor=base.scene['contact_forces']
        self.names=[n for n in self.robot.body_names if n.endswith('_foot')]
        if len(self.names)!=4:raise ValueError('E1 requires exactly four named feet')
        self.ids=[self.robot.body_names.index(n) for n in self.names]
        self.sids=[self.sensor.body_names.index(n) for n in self.names]
        self.metrics=EpisodeMetrics(base.num_envs,self.names,condition)
        self.flat=None
        if os.environ.get('E1_B2_METRICS')=='1':
            from e1_b2_metrics import FlatPitchGeometry
            self.flat=FlatPitchGeometry(base,geometry)
        self.original=base._reset_idx
        def before_reset(env,ids,*args,**kwargs):
            self.sample(ids,terminal=True)  # capture final pose/contact BEFORE reset changes them
            return self.original(ids,*args,**kwargs)
        base._reset_idx=types.MethodType(before_reset,base)
    def edges(self):
        slots,lower=self.geometry.for_envs(self.base);o=self.base.scene.env_origins
        return slots[:,1,0]+o[:,0],slots[:,1,1]+o[:,0],lower[:,1]+o[:,2]
    def sample(self,ids=None,terminal=False):
        b=self.base;d=self.robot.data
        ids=torch.arange(b.num_envs,device=b.device) if ids is None else ids
        if not len(ids):return
        w,x,y,z=d.root_quat_w.unbind(-1)
        pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1));roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        if self.flat is not None:self.flat.sample(ids,pitch)
        contact=self.sensor.data.net_forces_w[:,self.sids].norm(dim=-1)>15.
        time=torch.full((b.num_envs,1),b.common_step_counter*b.step_dt,device=b.device)
        packed=torch.cat((time,d.root_pos_w[:,[0,2]],pitch[:,None],roll[:,None],d.body_pos_w[:,self.ids].flatten(1),contact.float(),self.switch.first_step[:,None]),-1)
        near,far,deck=self.edges();data=packed[ids].detach().cpu().numpy()
        geo=torch.stack((near,far,deck),-1)[ids].detach().cpu().numpy()
        enabled=self.family()[ids].detach().cpu().tolist()
        timeouts=b.termination_manager.time_outs[ids].detach().cpu().tolist() if terminal else [False]*len(ids)
        for k,e in enumerate(ids.tolist()):
            if not enabled[k]:continue
            if self.metrics.geometry[e] is None:self.metrics.geometry[e]=tuple(float(v) for v in geo[k])
            self.metrics.rows[e].append(data[k].copy())
            if terminal:self.metrics.close(e,'timeout' if timeouts[k] else 'terminated')
    def close(self):self.base._reset_idx=self.original

@torch.inference_mode()
def run(env,teacher,student,attention,transport,args,agent_cfg,env_cfg,teacher_path,iteration):
    assert args.e1_condition == "a_student_gate"
    assert env.num_envs == 32 and abs(env.unwrapped.step_dt-.01)<1e-9
    from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
    from gd_lab.mdp.platform_gap_terms import boarding_family_mask
    base=env.unwrapped;n=base.num_envs;device=base.device;dt=base.step_dt
    student.eval()
    for p in student.parameters():p.requires_grad_(False)
    teacher.actor.eval()
    fix_velocity(base.command_manager.get_term('base_velocity'),args.replay_vx)
    obs,_=env.reset()
    from gain_control import freeze_gain_resampling
    gain_control=freeze_gain_resampling(base)
    initial_environment=environment_metadata(base)
    driver=E1V3Driver(student,n,device)
    episodes=driver.bank.episode
    def pose():
        return torch.cat((pose_xyyaw(base),base.scene['robot'].data.root_quat_w),-1).detach().clone()
    switch=LatentSwitch(n,device);geometry=GapTileGeometry(base)
    rec=Recorder(base,switch,args.e1_condition,geometry,lambda:boarding_family_mask(base))
    # Save comparable initial state evidence across independently launched conditions.
    initial=torch.cat((base.scene['robot'].data.root_state_w,base.scene['robot'].data.joint_pos),-1).cpu().numpy()
    initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
    initial_terrain={'levels':base.scene.terrain.terrain_levels.cpu().tolist(),'types':base.scene.terrain.terrain_types.cpu().tolist()}
    next_capture=transport.next_capture(base.common_step_counter);base._vrl_camera_capture_step=next_capture
    start=int(base.common_step_counter);steps=round(args.replay_seconds/dt);switches=[];max_equivalence=0.
    try:
        for k in range(steps):
            step=int(base.common_step_counter)
            if step==next_capture:
                stamps=getattr(base,'_vrl_camera_snapshot_steps',None)
                if stamps is None or not (stamps==step).all():
                    raise RuntimeError('scheduled capture did not produce fresh 4-view images')
                transport.capture(step,(base._vrl_camera_snapshot[0].clone(),driver.bank.episode.clone(),pose()))
                next_capture=transport.next_capture(step);base._vrl_camera_capture_step=next_capture
            for packet in transport.receive(step):
                driver.deliver(packet,step)
            packed=teacher._actor_input(obs,inference=True)
            actions,latent,effective=driver.action(teacher.actor,packed[:,:-32],obs['policy'],pose(),step)
            rec.sample()
            executed=actions.detach().clone()
            if env.clip_actions is not None:executed=executed.clamp(-env.clip_actions,env.clip_actions)
            obs,_,done,_=env.step(executed)
            reset=done.reshape(-1).bool()
            driver.after_step(executed,reset,int(base.common_step_counter))
            switch.reset(reset)
            if k == 0 or (k + 1) % 100 == 0:
                print(f"[E1] rollout step={k + 1}/{steps} sim_step={int(base.common_step_counter)}", flush=True)
        if environment_metadata(base)!=initial_environment:raise RuntimeError('actuator gain changed after freeze')
        records=rec.metrics.finish()
        def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result={'schema':'update26_e1_v1','condition':args.e1_condition,'seed':agent_cfg.seed,'num_envs':n,'seconds':steps*dt,
          'student_arch':'gast_belief_v3','student_size':'small','student_updates':iteration,'student_iteration':iteration,'student_sha256':digest(args.student_resume),'teacher_sha256':digest(teacher_path),
          'gain_control':gain_control,'initial_environment':initial_environment,'initial_state_sha256':initial_hash,'initial_terrain':initial_terrain,'environment':environment_metadata(base),
          'camera_transport':transport.config.manifest(dt),'transport_counts':{'captured':transport.captured,'dropped':transport.dropped,'delivered':transport.delivered},
          'max_teacher_action_equivalence_error':max_equivalence,'switches':switches,'approaches':records,
          'definitions':{'early_m':1.,'late_m':.3,'switch':'latch at exact pre-action step boundary through episode end; reset clears it',
             'baseline':'v3 BeliefBank: arrival publish, policy-rate belief, capture age <30 ticks, raw*effective gate once',
             'gate1':'last raw head, quality and age gate removed; zero until first valid packet',
             'teacher_history':'clean gast_history evaluated at policy rate, native history capture cadence unchanged',
             'pitch_flat':'per-episode base x 1.3–2.5m before the forward gap; null if absent',
             'geometry':'exact GapTileGeometry forward/right slot; gap-family envs only; lower deck used as height zero',
             'metrics':'UPDATE21 first crossing swings; foot centers; contact force norm>15N; clean=base beyond far+.8 and upright and no center below deck over slot',
             'termination':'captured before auto-reset; window end censored; no full RolloutLog'},'start_step':start}
        if rec.flat is not None:result['update34_flat_pitch']=rec.flat.result()
        path=Path(args.replay_out)
        with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        print('[E1] COMPLETE',path,'approaches',len(records),flush=True)
    finally:rec.close()
