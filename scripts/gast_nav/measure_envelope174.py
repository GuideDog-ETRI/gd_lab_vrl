"""Frozen3906 command-response measurement. LocalGPU0 only, no optimizer."""
import argparse,os,json,math,time
from pathlib import Path
ROOT=Path('/home/user/gd_project/archive/gd_lab_vrl_worktrees/gast_nav_v1')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
a.headless=True;a.device='cuda:0'
a.kit_args=' '.join(f'--/app/tokens/{k}={a.output}/runtime/{k}' for k in ('cache','data','logs'))
app=AppLauncher(a).app
import faulthandler,signal
stack_file=open(a.output/'stack.log','ab',buffering=0)
faulthandler.enable(file=stack_file,all_threads=True)
faulthandler.register(signal.SIGUSR2,file=stack_file,all_threads=True)
try:
 import torch,gymnasium as gym
 from tensordict import TensorDict
 import gd_lab.gast_nav
 from gd_lab.gast_nav.env_cfg import GastNavEnvCfg
 from gd_lab.gast_nav.obs_terms import world_to_sample
 from gd_lab.gast.teacher import GastRayIrActorCritic
 from isaaclab.terrains import MeshPlaneTerrainCfg
 cfg=GastNavEnvCfg(nav_stage='A0');cfg.scene.num_envs=64;cfg.sim.device='cuda:0'
 gen=cfg.scene.terrain.terrain_generator;gen.num_rows=2;gen.num_cols=4
 from gd_lab.gast_nav.fixture174 import configure_fixture174
 import yaml
 from gd_lab.gast_nav.scene_config import load_scene_config
 from gd_lab.gast_nav.curriculum167 import easy_scene_settings
 limits=yaml.safe_load((ROOT/'configs/gast_nav/curriculum_C.yaml').read_text())['scene_limits']
 scene_file=a.output/'easy_scenes.yaml';scene_file.write_text(yaml.safe_dump(easy_scene_settings(load_scene_config(),limits)))
 configure_fixture174(cfg.scene.terrain,64,scene_file,seed=174)
 cfg.nav_settings['primitive_c169']=True
 cfg.nav_stage_settings['moving_cylinder']=False
 cfg.scene.nav_cylinder.spawn.collision_props.collision_enabled=False
 cfg.scene.terrain.max_init_terrain_level=0
 cfg.rewards.stair_handle_disturbance.weight=0.
 for group in (cfg.rewards,cfg.curriculum):
  for name in list(vars(group)):
   if not name.startswith('_'):setattr(group,name,None)
 for name in list(vars(cfg.terminations)):
  if not name.startswith('_'):setattr(cfg.terminations,name,None)
 print('ENV CREATE',flush=True)
 env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
 print('INITIAL RESET',flush=True)
 raw,_=env.reset();print('RESET OK',flush=True);obs=TensorDict(raw,batch_size=[64],device='cuda:0')
 fixed=GastRayIrActorCritic(obs=obs,obs_groups={'policy':['policy'],'critic':['critic']},num_actions=12,
  actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',history_length=5,policy_term_dims=[3,3,3,12,12,12,1],velocity_target_slice=(45,48),actor_history_steps=4,height_scan_start=110,adaboot_enabled=False).cuda()
 checkpoint=ROOT/'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
 fixed.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model_state_dict'],strict=True)
 fixed.eval();fixed.requires_grad_(False);env.nav.fixed_policy=fixed
 # Isolated open-loop command-response fixture, NOT NAV acceptance evaluation.
 # Remove map/FSM work; only frozen controller actions drive a flat-floor robot.
 env.nav.prepare=lambda raw:raw
 command=env.command_manager.get_term('base_velocity')
 results=[];dt=env.step_dt
 with torch.inference_mode():
  for seed in (17421,17422,17423):
   for mode,value in [('NAV',.2),('NAV',.4),('NAV',.6),('TURN',-.3),('TURN',-.6)]:
    print(f'TRIAL {seed} {mode} {value}',flush=True)
    raw,_=env.reset(seed=seed);env.nav.legacy[:]=False;env.nav.controller[:]=0
    env.nav.mode[:]=1 if mode=='NAV' else 2
    command.is_heading_env[:]=False
    # Additional posture domains: open flat and actual ascending/descending treads.
    kinds=[t.kind for t in env.nav.index.lookup(range(64))]
    state=env.scene['robot'].data.root_state_w.clone()
    for i,tile in enumerate(env.nav.index.lookup(range(64))):
     if tile.kind=='nav_corridor' and i%8==0:
      state[i,:2]=env.scene.env_origins[i,:2]+state.new_tensor([-8.,0.]);kinds[i]='flat'
     if tile.kind=='nav_stair_landing':
      solids=cfg.scene.terrain.terrain_generator.sensor_solids169[(i//8,i%8)]
      steps=sorted([b for b in solids if b.kind=='step'],key=lambda b:b.center[0])
      part=(i//4)%3
      candidates=[b for b in steps if b.center[0]<0] if part==0 else ([b for b in steps if b.center[0]>0] if part==2 else [b for b in steps if b.center[0]==0])
      b=candidates[(seed+i)%len(candidates)]
      state[i,0]=env.scene.env_origins[i,0]+b.center[0];state[i,1]=env.scene.env_origins[i,1]
      state[i,2]=env.scene.env_origins[i,2]+b.center[2]+b.half[2]+env.scene['robot'].data.default_root_state[i,2]
      kinds[i]+='/'+('up','landing','down')[part]
    env.scene['robot'].write_root_state_to_sim(state)

    delay=math.ceil((.04+dt)/dt);ramp=math.ceil(abs(value)/(.5 if mode=='NAV' else 1.)/dt)
    starts=torch.randint(180,301,(64,),device='cuda:0');starts[torch.arange(64,device='cuda:0')%3==0]=100;total=301+delay+ramp+150;record=[];bad=torch.zeros(64,dtype=torch.bool,device='cuda:0')
    for tick in range(total):
     if tick%100==0:print(f'tick={tick}',flush=True)
     elapsed=(tick-starts-delay+1).clamp_min(0)*dt
     level=(abs(value)-elapsed*(.5 if mode=='NAV' else 1.)).clamp_min(0)*math.copysign(1.,value)
     if tick<100:level.zero_()
     vel=torch.zeros(64,3,device='cuda:0');vel[:,0 if mode=='NAV' else 2]=level
     env.nav.velocity.copy_(vel);env.nav.controller[:]=0;command.vel_command_b.copy_(vel)
     raw['policy'][:,30:45].reshape(64,5,3)[:,-1]=vel*torch.as_tensor(cfg.observations.policy.velocity_commands.scale,device=vel.device)
     action=fixed.act_inference(TensorDict(raw,batch_size=[64],device='cuda:0'))
     raw,reward,terminated,truncated,extra=env.step(action)
     d=env.scene['robot'].data
     pose=torch.cat((d.root_pos_w,d.heading_w[:,None]),-1)
     bad|=terminated|truncated
     if tick>=0:
      record.append(dict(pos=d.body_pos_w.clone(),quat=d.body_quat_w.clone(),base_pose=pose.clone(),
       base_local=world_to_sample(d.body_pos_w,pose),joint_pos=d.joint_pos.clone(),
       speed=d.root_lin_vel_b[:,0].clone(),omega=d.root_ang_vel_b[:,2].clone(),command=level.clone()))
    payload={k:torch.stack([r[k] for r in record]).cpu() for k in record[0] }
    payload.update(invalid=bad.cpu(),starts=starts.cpu(),kinds=kinds,start_pose=payload['base_pose'][starts.cpu()-1,torch.arange(64)],
      body_names=env.scene['robot'].body_names,dt=dt,seed=seed,mode=mode,value=value,delay_s=delay*dt,angular_deceleration_provisional=1.,optimizer_steps=0)
    filename=f'{mode}_{abs(value):.1f}_seed{seed}.pt';torch.save(payload,a.output/filename)
    item=dict(file=filename,seed=seed,mode=mode,value=value,invalid=int(bad.sum()),ticks=len(record));results.append(item)
    print(json.dumps(item),flush=True)
 (a.output/'measurement.json').write_text(json.dumps(dict(status='completed',trials=results,optimizer_steps=0,fixture='mixed easy C, random start ticks; all trials retained including no-stop; diagnostic temporary, NOT deployment assurance'),indent=2))
 env.close()
except BaseException:
 import traceback
 traceback.print_exc()
 os._exit(1)
finally:app.close()
