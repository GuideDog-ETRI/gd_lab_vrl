"""GAST-Nav task configuration derived from the unchanged 3906 environment."""
from pathlib import Path
import yaml
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from .person_contact import person_contact_termination, person_invalid_termination
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from gd_lab.gast.tasks import GastRayIrV21Gap50TeacherCfg
from .terrains import make_isaac_cfg
from .scene_adapter import generator_class
from .obs_terms import isaac_sensor_cfg
from .isaac_terms import (NavJointPositionAction,NavVelocityCommand,SummedHandleDisturbance,
                          biased_gravity,masked_term,nav_reward)
ROOT=Path(__file__).resolve().parents[3]

def settings(stage):
    base=yaml.safe_load((ROOT/'configs/gast_nav/environment.yaml').read_text())
    phase=yaml.safe_load((ROOT/f'configs/gast_nav/stage_{stage}.yaml').read_text())
    return base,phase

@configclass
class GastNavEnvCfg(GastRayIrV21Gap50TeacherCfg):
    nav_stage: str='B'
    nav_smoke: bool=False
    def __post_init__(self):
        super().__post_init__()
        base,stage=settings(self.nav_stage)
        self.nav_settings=base;self.nav_stage_settings=stage
        self.sim.dt=base['physics_dt'];self.decimation=base['policy_decimation']
        gen=self.scene.terrain.terrain_generator
        gen.class_type=generator_class();gen.use_cache=False
        gen.size=(16.,12.);gen.num_cols=80
        gen.difficulty_range=tuple(stage['difficulty_range'])
        total=sum(v.proportion for v in gen.sub_terrains.values())
        for cfg in gen.sub_terrains.values():cfg.proportion=cfg.proportion/total*base['scene_mix']['legacy']
        for kind,p in base['scene_mix'].items():
            if kind!='legacy':gen.sub_terrains[kind]=make_isaac_cfg(kind,size=gen.size,proportion=p)
        self.scene.nav_obstacles=isaac_sensor_cfg(base['sensors'],prim_path='{ENV_REGEX_NS}/Robot/trunk',
                                                 mesh_prim_path='/World/ground',update_period=base['capture_period_s'])
        self.scene.replicate_physics=False  # per-env cylinder radii must remain independent
        self.scene.nav_cylinder=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/NavCylinder',
            spawn=sim_utils.CylinderCfg(radius=.35,height=1.7,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-100.)))
        for link in base['person_contact']['robot_links']:
            setattr(self.scene,'person_contact_'+link,ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Robot/'+link,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/NavCylinder'],
                update_period=self.sim.dt,history_length=self.decimation,
                max_contact_data_count_per_prim=base['person_contact']['max_contacts_per_prim']))
        self.terminations.person_contact=TerminationTermCfg(func=person_contact_termination)
        self.terminations.person_invalid=TerminationTermCfg(func=person_invalid_termination)
        self.actions.joint_pos.class_type=NavJointPositionAction
        self.commands.base_velocity.class_type=NavVelocityCommand
        self.observations.policy.projected_gravity.func=biased_gravity
        self.rewards.stair_handle_disturbance.func=SummedHandleDisturbance
        for name in ('track_lin_vel_xy_exp','track_ang_vel_z_exp'):
            term=getattr(self.rewards,name,None)
            if term is not None:term.func=masked_term(term.func,'legacy')
        if getattr(self.rewards,'stair_stall',None) is not None:
            self.rewards.stair_stall.func=masked_term(self.rewards.stair_stall.func,'stall')
        self.rewards.nav_total=RewardTermCfg(func=nav_reward,weight=1.)
