"""Paired nominal flat-ground, zero-command teacher/student diagnostic."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--bundle', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--velocities', type=float, nargs='+', default=[0.0])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime
root = prepare_vrl_runtime(args)
app = AppLauncher(args)
verify_vrl_runtime(root)

import gymnasium as gym
import torch
import gd_lab
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from gd_lab.teachers.cvtt.agent_cfg import DreamwaqVrlRunnerCfg
from gd_lab.rl.runner import DreamwaqRunner
from gd_lab.students.rvld.model import CameraPerceptionEncoder
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
from gd_lab.core.camera_contract import camera_contract_for_policy
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking

bundle = Path(args.bundle)
cfg = parse_env_cfg('Gd-Vrl-Rbq10-Dreamwaq-VisionPlay-v0', device=args.device, num_envs=1)
cfg.seed = 42
cfg.sim.dt = .005
cfg.decimation = 2
cfg.episode_length_s = 60
cfg.scene.terrain.terrain_type = 'plane'
cfg.scene.terrain.terrain_generator = None
cfg.scene.terrain.max_init_terrain_level = None
cfg.rewards.platform_gap_crossing = None
cfg.curriculum.terrain_levels = None
for name in ('rbq_hip', 'rbq_knee'):
    cfg.scene.robot.actuators[name].stiffness = 123.39 if name == 'rbq_hip' else 127.77
    cfg.scene.robot.actuators[name].damping = 2.4
for name in ('physics_material', 'add_base_mass', 'base_com', 'randomize_base_inertia', 'randomize_actuator_gains', 'push_robot'):
    setattr(cfg.events, name, None)
cfg.events.randomize_payload.params['payload_masses'] = (6.0,)
cfg.events.randomize_payload.params['payload_probabilities'] = (1.0,)
cfg.events.randomize_payload.params['payload_position_range'] = {'x': (0.,0.), 'y': (0.,0.), 'z': (.07,.07)}
cfg.events.reset_robot_joints.params['position_range'] = (1.,1.)
cfg.events.reset_base.params['pose_range'] = {'x': (0.,0.), 'y': (0.,0.), 'yaw': (0.,0.)}
c = cfg.commands.base_velocity
c.heading_command = False
c.rel_heading_envs = 0.
c.rel_standing_envs = 1.
c.ranges.lin_vel_x = c.ranges.lin_vel_y = c.ranges.ang_vel_z = (0.,0.)
c.pulse_prob = 0.
configure_vrl_cameras(cfg)
env = gym.make('Gd-Vrl-Rbq10-Dreamwaq-VisionPlay-v0', cfg=cfg)
ensure_prev_prev_action_tracking(env.unwrapped)
acfg = DreamwaqVrlRunnerCfg()
acfg.device = args.device
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
runner = DreamwaqRunner(env, acfg.to_dict(), log_dir=None, device=args.device)
runner.load(str(bundle/'teacher/model_3700.pt'), load_optimizer=False)
teacher = runner.alg.policy
teacher.eval()
student = CameraPerceptionEncoder(num_cameras=4, gru_hidden_dim=64, latent_dim=32).to(args.device)
ckpt = torch.load(bundle/'student/perception_12400.pt', map_location=args.device, weights_only=False)
from gd_lab.core.camera_contract import same_camera_contract  # noqa: E402
assert same_camera_contract(ckpt['camera_contract'], camera_contract_for_policy(cfg.camera_profile, .01))
student.load_state_dict(ckpt['model'])
student.eval()
results = {}
import itertools
for mode, vx in itertools.product(('teacher', 'student'), args.velocities):
    label = f'{mode}_vx{vx:g}'
    command_cfg = env.unwrapped.command_manager.get_term('base_velocity').cfg
    command_cfg.ranges.lin_vel_x = (vx, vx)
    command_cfg.rel_standing_envs = 1.0 if vx == 0 else 0.0
    with torch.inference_mode():
        env.seed(42)
        env.reset()
        obs = env.get_observations()
    hidden = student.init_hidden(1, args.device)
    latent = torch.zeros((1,32), device=args.device)
    rows=[]
    traces=[]
    resets=0
    with torch.inference_mode():
        for step in range(args.steps):
            raw=env.unwrapped
            cmd=raw.command_manager.get_command('base_velocity')
            assert abs(float(cmd[0,0])-vx) < 1e-6 and float(cmd[0,1:].abs().max()) < 1e-7, cmd
            if mode == 'teacher':
                actions=teacher.act_with_terrain_latent(obs, teacher.terrain_latent(obs))
            else:
                if step % 8 == 0:
                    frames=raw._vrl_camera_snapshot[0]
                    latent,hidden=student(frames,hidden)
                actions=teacher.act_with_terrain_latent(obs,latent)
            before_q = raw.scene['robot'].data.joint_pos.clone()
            obs,_,dones,_=env.step(actions)
            robot=raw.scene['robot'].data
            term = raw.action_manager.get_term('joint_pos')
            traces.append(torch.cat((actions[0], term.processed_actions[0],
                                     robot.joint_pos_target[0], before_q[0],
                                     robot.joint_pos[0], robot.joint_vel[0])).cpu())
            if step>=200:
                rows.append([float(robot.joint_vel.abs().mean()),float(robot.root_pos_w[0,2]),float(robot.root_lin_vel_w[0,2]),float(robot.root_ang_vel_b.norm()),float(actions.abs().mean())])
            if dones.any():
                resets+=1
                hidden.zero_();latent.zero_()
            if step%250==0:
                print('PROGRESS',label,step,'qd',float(robot.joint_vel.abs().mean()),flush=True)
    values=torch.tensor(rows)
    trace=torch.stack(traces)
    metrics={}
    for name, sl in [('raw_action',slice(0,12)),('pre_limit_target',slice(12,24)),('applied_target',slice(24,36)),('actual_position',slice(48,60))]:
        x=trace[200:,sl]
        delta=x.diff(dim=0).abs()
        metrics[name]={'max_abs':x.abs().max().item(),'max_step':delta.max().item(),
                       'p99_step':torch.quantile(delta.flatten(),.99).item(),
                       'steps_over_0_1':int((delta.max(dim=1).values>.1).sum())}
    metrics['tracking_error_mean_rad']=(trace[200:,24:36]-trace[200:,48:60]).abs().mean().item()
    results[label]={'mean_abs_joint_velocity':values[:,0].mean().item(),'height_mean':values[:,1].mean().item(),'height_std':values[:,1].std().item(),'vertical_velocity_rms':values[:,2].square().mean().sqrt().item(),'resets':resets,'samples':len(rows),'action_metrics':metrics}
    trace_path=Path(args.output).with_suffix('')
    torch.save({'trace':trace,'columns':['raw_action','pre_limit_target','applied_target','q_before','q_after','qd_after'],
                'joint_names':raw.scene['robot'].joint_names,'dt':.01,'command':[vx,0,0]},str(trace_path)+'_'+label+'.pt')
    print('RESULT',label,json.dumps(results[label]),flush=True)
    Path(args.output).write_text(json.dumps({'conditions':{'steps':args.steps,'dt':.01,'payload_kg':6,'seed':42,'warmup_steps':200,'terrain':'plane','velocities':args.velocities},'results':results},indent=2))
env.close()
app.app.close()
