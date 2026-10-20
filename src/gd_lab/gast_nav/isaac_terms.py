"""Isaac-only subclasses/hooks. Import after AppLauncher."""
import functools
import torch
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.math import quat_apply
from gd_lab.mdp.gap_stair_v21 import V21VelocityCommand
from gd_lab.mdp.gap_stair_v2 import StairHandleDisturbance

class NavJointPositionAction(JointPositionAction):
    def apply_actions(self):
        if getattr(self._env,'nav',None) is not None:self._env.nav.physics_step()
        super().apply_actions()

class NavVelocityCommand(V21VelocityCommand):
    def _update_command(self):
        nav=getattr(self._env,'nav',None)
        if nav is not None:
            self.is_heading_env &= nav.legacy
        super()._update_command()
        if nav is not None:
            self.is_heading_env &= nav.legacy
            self.vel_command_b[~nav.legacy]=nav.velocity[~nav.legacy]
            self.vel_command_b[nav.controller!=0]=0

class _HandleProxy:
    def __init__(self,robot,env):self.robot=robot;self.env=env
    def __getattr__(self,key):return getattr(self.robot,key)
    def set_external_force_and_torque(self,forces,torques,*,positions,body_ids):
        nav=getattr(self.env,'nav',None)
        if nav is None:return
        robot=self.robot;q=robot.data.body_quat_w[:,body_ids]
        force=quat_apply(q,forces);torque=quat_apply(q,torques)
        point=robot.data.body_pos_w[:,body_ids]+quat_apply(q,positions)
        torque=torque+torch.cross(point-robot.data.body_com_pos_w[:,body_ids],force,dim=-1)
        nav.forces.force.zero_();nav.forces.torque.zero_()
        nav.forces.force[:,body_ids]=force;nav.forces.torque[:,body_ids]=torque

class SummedHandleDisturbance(StairHandleDisturbance):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.robot=_HandleProxy(self.robot,env)


def biased_gravity(env):
    raw=env.scene['robot'].data.projected_gravity_b
    nav=getattr(env,'nav',None)
    return raw if nav is None else nav.inertia.gravity_bias(raw)


def masked_term(original,mask_name):
    def mask(env,value):
        nav=getattr(env,'nav',None)
        if nav is None:return value
        if mask_name=='legacy':return value*nav.legacy
        return value*nav.stall_enabled
    if isinstance(original,type):
        class Masked(original):
            @functools.wraps(original.__call__)
            def __call__(self,env,*args,**kw):return mask(env,super().__call__(env,*args,**kw))
        return Masked
    @functools.wraps(original)
    def wrapped(env,*args,**kw):return mask(env,original(env,*args,**kw))
    return wrapped


def nav_reward(env):
    nav=getattr(env,'nav',None)
    if nav is None:return torch.zeros(env.num_envs,device=env.device)
    total=nav.compute_rewards()
    nav.reward_weighted_rm=nav.reward_weighted  # the RewardManager tick (prepare() recomputes after resets)
    return total
