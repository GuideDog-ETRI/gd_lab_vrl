"""CPU-testable pre-policy control bridge, ready for an Isaac environment adapter.

LandingObservation must come from measured terrain/contact evidence. This module
never derives landing detection or phase from privileged SceneTileMeta bounds.
"""
from dataclasses import dataclass, replace
import math
from .command_generator import CommandGenerator
from .commands import policy_features
from .landing_fsm import LandingFSM
from .supervisor import Supervisor
from . import schema

@dataclass(frozen=True)
class ControlOutput:
    controller: str
    fallback_action: str
    reason: str
    velocity: tuple
    features: list
    phase: str
    effective_go: bool
    intervention_started: bool
    fsm: dict

class ControlRuntime:
    def __init__(self,tiles,environment_cfg,runtime_cfg,*,seed):
        self.commands=CommandGenerator(tiles,environment_cfg,runtime_cfg['command'],seed=seed)
        self.fsm=[LandingFSM(runtime_cfg['landing']) for _ in tiles]
        self.supervisors=[Supervisor(runtime_cfg['supervisor']) for _ in tiles]
        self.previous=['NAV_POLICY']*len(tiles)
        self.cfg=environment_cfg
        self.mode=environment_cfg.get("supervisor_mode", "enforce")
        if self.mode not in ("shadow", "enforce"):raise ValueError("supervisor_mode")

    def reset(self,env_ids,*,now_ms,yaws,tiles=None):
        ids=[int(i) for i in env_ids]
        self.commands.reset(ids,now_ms=now_ms,yaws=yaws,tiles=tiles)
        for i in ids:
            self.fsm[i].reset()
            self.supervisors[i].turn_states.clear()
            self.previous[i]='NAV_POLICY'

    def before_policy(self,i,*,now_ms,dt,xy,yaw,health,landing_observation,odom_epoch,legacy_velocity,route_info=None,unknown_wait=False):
        i=int(i);supervisor=self.supervisors[i]
        decision=supervisor.select(health,simulation=True)
        shadow=self.mode=='shadow'
        # Keep invalid commands and failed inference actionable. Sensor/risk
        # decisions remain available in env metrics, not actuator selection.
        if shadow and decision.health in ('SUPERVISOR_RISK','TERRAIN_STALE','POSE_STALE'):
            decision=replace(decision,controller='NAV_POLICY',fallback_action='',decelerate=False)
        if shadow:unknown_wait=False
        hold=decision.controller!='NAV_POLICY'
        o=landing_observation
        fsm=self.fsm[i]
        stable=(o.four_contacts and not o.slipping and abs(o.speed_mps)<fsm.cfg['stable_v_mps']
                and abs(o.yaw_rate_radps)<fsm.cfg['stable_w_radps'])
        # Update command and handshake exactly once per policy tick. FSM then
        # gates the result, avoiding a second adapter.tick advancing handshake.
        cmd,applied,_,_=self.commands.tick(i,now_ms=now_ms,dt=dt,xy=xy,stable=stable,hold=hold,
                                          phase='APPROACH' if shadow else fsm.phase,legacy=legacy_velocity,route_info=route_info,return_features=False)
        state=(dict(phase='APPROACH',effective_go=applied['effective_go'],hold_reason='',dwell_progress=0.) if shadow else fsm.tick(now_ms=now_ms,dt=dt,command=cmd,observation=o,hold=hold or unknown_wait,
                       sensor_timeout=decision.health in ('TERRAIN_STALE','POSE_STALE'),odom_epoch=odom_epoch))
        reverse=supervisor.observe_turn(route_epoch=cmd.route_epoch,manoeuvre_id=cmd.manoeuvre_id,
                                        active_mode=applied['mode'],yaw=yaw,yaw_rate=o.yaw_rate_radps,dt=dt)
        invalid=bool(applied['hold_reason']) and applied['hold_reason']!='HOLD'
        reason=decision.health if hold else state['hold_reason'] or applied['hold_reason']
        controller=decision.controller
        fallback=decision.fallback_action
        if controller=='NAV_POLICY' and ((reverse and not shadow) or invalid):
            controller='HOLD_3906';fallback='HOLD';reason='TURN_REVERSE' if reverse and not shadow else applied['hold_reason']
        hold=controller!='NAV_POLICY'
        if unknown_wait and not hold:
            reason='UNKNOWN_WAIT'
        if hold or unknown_wait or not state['effective_go']:
            applied=dict(applied,velocity=(0.,0.,0.),effective_go=False)
        elif state['phase']=='PRE_BRAKE' and applied['mode']=='NAV':
            braking=math.sqrt(2*self.cfg['commands']['brake_deceleration_mps2']*cmd.route_hint.dist_to_next_m)
            applied=dict(applied,velocity=(min(cmd.v_cap,braking),0.,0.))
        started=controller=='HOLD_3906' and self.previous[i]!='HOLD_3906'
        self.previous[i]=controller
        features=policy_features(applied,self.commands.adapters[i].c,state['phase'],state['dwell_progress'])
        return ControlOutput(controller,fallback,reason,applied['velocity'],features,state['phase'],
                             applied['effective_go'],started,state)


def zero_legacy_command_history(observations,*,term_dims,history_length,command_term=2):
    """Clone policy input and zero command history for fixed 3906 HOLD inference.

    Isaac concatenates history by term in the inherited 3906 policy group.
    Never mutate NAV observations or stored rollout samples.
    """
    out=observations.clone()
    start=sum(term_dims[:command_term])*history_length
    end=start+term_dims[command_term]*history_length
    if out['policy'].shape[-1]!=sum(term_dims)*history_length:
        raise ValueError('3906 term-major history layout mismatch')
    out['policy'][...,start:end]=0
    return out
