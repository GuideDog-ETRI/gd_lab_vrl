"""Pre-policy state controller selection only. No action proposal inspection or actuator writes."""
from dataclasses import dataclass
import math
from . import schema

@dataclass(frozen=True)
class HealthInput:
    terrain_age_ms: float
    obstacle_age_ms: float
    pose_age_ms: float
    infer_failed: bool = False
    nav_invalid: bool = False
    risk: bool = False

@dataclass(frozen=True)
class Decision:
    health: str
    controller: str | None
    fallback_action: str
    decelerate: bool = False

class Supervisor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.turn_states = {}

    def select(self, health, *, simulation):
        h = health
        if h.infer_failed:
            if simulation:
                controller = self.cfg['sim_infer_controller']
                if controller not in schema.CONTROLLERS: raise ValueError('invalid sim infer controller')
                return Decision('INFER_FAIL', controller, 'VISION_HOLD')
            return Decision('INFER_FAIL', 'DAMP', 'DAMP')
        if not all(math.isfinite(v) and v >= 0 for v in (h.terrain_age_ms, h.obstacle_age_ms, h.pose_age_ms)):
            return Decision('POSE_STALE', 'HOLD_3906', 'BLACKOUT', True)
        if h.nav_invalid: return Decision('NAV1_INVALID', 'HOLD_3906', 'HOLD')
        if h.pose_age_ms > self.cfg['pose_age_ms']: return Decision('POSE_STALE', 'HOLD_3906', 'BLACKOUT', True)
        if max(h.terrain_age_ms, h.obstacle_age_ms) > self.cfg['terrain_age_ms']:
            return Decision('TERRAIN_STALE', 'HOLD_3906', 'BLACKOUT', True)
        if h.risk: return Decision('SUPERVISOR_RISK', 'HOLD_3906', 'HOLD')
        return Decision('OK', 'NAV_POLICY', '')

    def observe_turn(self, *, route_epoch, manoeuvre_id, active_mode, yaw, yaw_rate, dt):
        if not all(math.isfinite(v) for v in (yaw, yaw_rate, dt)) or dt < 0: raise ValueError('turn sample')
        key = (route_epoch, manoeuvre_id)
        state = self.turn_states.setdefault(key, {'yaw': yaw, 'positive': 0., 'rate_s': 0., 'was_turn': False})
        delta = math.atan2(math.sin(yaw - state['yaw']), math.cos(yaw - state['yaw']))
        state['yaw'] = yaw
        if active_mode != 'TURN':
            state['rate_s'] = 0.; state['was_turn'] = False; return False
        if not state['was_turn']: delta = 0.
        state['was_turn'] = True
        if delta > self.cfg['yaw_deadband_rad']: state['positive'] += delta
        state['rate_s'] = state['rate_s'] + dt if yaw_rate > self.cfg['reverse_rate_radps'] else 0.
        return (state['positive'] >= math.radians(self.cfg['reverse_angle_deg']) or
                state['rate_s'] >= self.cfg['reverse_rate_duration_s'])

@dataclass(frozen=True)
class Horizon:
    future_s: float
    person_inflation_m: float
    residual_margin_m: float
    fallback: bool

def predict_horizon(*, map_age_s, compute_actuation_s, stopping_s, speed_mps, cfg):
    """Caller reconstructs sample->now from odom, then integrates future_s swept volume.
    No collision-free claim is returned; geometry/integration is supplied by the map side.
    """
    values = (map_age_s, compute_actuation_s, stopping_s, speed_mps)
    if not all(math.isfinite(v) and v >= 0 for v in values): raise ValueError('invalid horizon inputs')
    future = compute_actuation_s + stopping_s
    return Horizon(future, cfg['v_person_max'] * (map_age_s + future), cfg['residual_margin_m'],
                   map_age_s > cfg['max_map_age_s'] or speed_mps > cfg['max_speed_mps'])
