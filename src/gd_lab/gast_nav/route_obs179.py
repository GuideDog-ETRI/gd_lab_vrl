"""D-5 selected-route observation; no map, feasibility or reward oracle reads.

Caller uses the selected gdm route, current odometry and the same odometry frame.
Scene simulation: command_generator.route_hint(tile, root_xy - env_origin_xy).
NAV1 route_axis_yaw remains absolute odom yaw, never already body-relative.
"""
import math
from . import schema


def route_features179(hint, body_yaw, *, mode):
    """Return [sin(axis-body), cos(axis-body), signed geometric crosstrack m].

    LEGACY is exactly zero (unchanged legacy commands). TURN can omit a route;
    NAV requires a complete selected hint. Missing NAV lateral error must NOT be
    guessed zero: caller treats ValueError as NAV1_INVALID/HOLD before inference.
    c.y_off stays an independent existing condition, not subtracted here.
    """
    if mode not in schema.MODES:
        raise ValueError('ROUTE_MODE')
    if mode == 'LEGACY':
        return [0., 0., 0.]
    if hint is None or hint.route_lateral_error_m is None:
        if mode == 'TURN':
            return [0., 0., 0.]
        raise ValueError('NAV1_ROUTE_INPUT_MISSING')
    values=(body_yaw,hint.route_axis_yaw,hint.route_lateral_error_m)
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError('NAV1_ROUTE_INPUT_INVALID')
    yaw=hint.route_axis_yaw-body_yaw
    return [math.sin(yaw), math.cos(yaw), hint.route_lateral_error_m]
