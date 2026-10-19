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
 cfg=GastNavEnvCfg(nav_stage='A0');cfg.scene.num_envs=16;cfg.sim.device='cuda:0'
 gen=cfg.scene.terrain.terrain_generator;gen.num_rows=2;gen.num_cols=4
 gen.sub_terrains={'flat':MeshPlaneTerrainCfg(proportion=1.)};gen.difficulty_range=(0.,0.)
 cfg.scene.terrain.max_init_terrain_level=0
 cfg.rewards.stair_handle_disturbance.weight=0.
 for group in (cfg.rewards,cfg.curriculum):
  for name in list(vars(group)):
   if not name.startswith('_'):setattr(group,name,None)
 print('ENV CREATE',flush=True)
 env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
 print('INITIAL RESET',flush=True)
 raw,_=env.reset();print('RESET OK',flush=True);obs=TensorDict(raw,batch_size=[16],device='cuda:0')
 fixed=GastRayIrActorCritic(obs=obs,obs_groups={'policy':['policy'],'critic':['critic']},num_actions=12,
  actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',history_length=5,policy_term_dims=[3,3,3,12,12,12,1],velocity_target_slice=(45,48),actor_history_steps=4,height_scan_start=110,adaboot_enabled=False).cuda()
 checkpoint=ROOT/'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
 fixed.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model_state_dict'],strict=True)
 fixed.eval();fixed.requires_grad_(False);env.nav.fixed_policy=fixed
 # Live LEGACY command/control regression smoke; no learning or nav certificate.
 assert bool(env.nav.legacy.all()), 'expected all-LEGACY plane fixture'
 trace=[]
 with torch.inference_mode():
  for step in range(100):
   raw,_,_,_,extra=env.step(fixed.act_inference(obs))
   obs=TensorDict(raw,batch_size=[16],device='cuda:0')
   trace.append(dict(step=step,mode=env.nav.mode.tolist(),controller=env.nav.controller.tolist(),reasons=list(env.nav.reasons)))
   assert not bool(extra['nav_replan_requested'].any())
 (a.output/'result.json').write_text(json.dumps(dict(status=('passed' if all(all(m==0 for m in r['mode']) and all(c==0 for c in r['controller']) for r in trace) else 'failed'),trace=trace,envs=16,steps=100,legacy_mode_preserved=all(all(m==0 for m in r['mode']) for r in trace),controller_preserved=all(all(c==0 for c in r['controller']) for r in trace),spurious_replans=0,optimizer_steps=0,scope='flat LEGACY smoke; not stair/gap task acceptance'),indent=2))
 env.close()
except BaseException:
 import traceback
 traceback.print_exc()
 os._exit(1)
finally:app.close()
