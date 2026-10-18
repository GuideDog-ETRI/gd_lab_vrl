"""Simulator-free geometric helpers; all operations are in env-local metres."""
import math
import numpy as np
from .scene_types import route_point


def box_corners_xy(box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return [(box.center[0]+c*x-s*y, box.center[1]+s*x+c*y)
            for x,y in ((-box.half[0],-box.half[1]),(box.half[0],-box.half[1]),
                        (box.half[0],box.half[1]),(-box.half[0],box.half[1]))]


def point_box_distance(xy, box):
    """Signed 2D distance to the actual yaw-oriented box boundary."""
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    dx, dy = xy[0]-box.center[0], xy[1]-box.center[1]
    x, y = abs(c*dx+s*dy)-box.half[0], abs(-s*dx+c*dy)-box.half[1]
    return math.hypot(max(x, 0.), max(y, 0.))+min(max(x, y), 0.)


def floor_height(meta, xy):
    """Critic/oracle-only top of local step solids; never a sensor support seed."""
    z = 0.
    if meta.floor_z_at == 'stairs':
        for box in meta.obstacles:
            if box.kind != 'step': continue
            c, s = math.cos(box.yaw), math.sin(box.yaw)
            dx, dy = xy[0]-box.center[0], xy[1]-box.center[1]
            if abs(c*dx+s*dy) <= box.half[0] and abs(-s*dx+c*dy) <= box.half[1]:
                z = max(z, box.center[2]+box.half[2])
    return z


def line_box_interval(point, direction, box, eps):
    """Infinite 2D line / closed OBB intersection parameter interval or None."""
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    px, py = point[0]-box.center[0], point[1]-box.center[1]
    origin = (c*px+s*py, -s*px+c*py)
    ray = (c*direction[0]+s*direction[1], -s*direction[0]+c*direction[1])
    low, high = -math.inf, math.inf
    for o, d, h in zip(origin, ray, box.half[:2]):
        if abs(d) <= eps:
            if abs(o) > h+eps: return None
        else:
            a, b = (-h-o)/d, (h-o)/d
            low, high = max(low, min(a,b)), min(high, max(a,b))
            if low > high: return None
    return low, high


def free_cross_section(point, tangent, nominal_width, obstacles, floor_z, height, eps):
    """Free lateral intervals inside the nominal route corridor, subtracting OBBs."""
    free = [(-nominal_width/2, nominal_width/2)]
    normal = (-tangent[1], tangent[0])
    for box in obstacles:
        if box.kind == 'step' or box.center[2]+box.half[2] <= floor_z+eps or box.center[2]-box.half[2] >= floor_z+height:
            continue
        interval = line_box_interval(point, normal, box, eps)
        if interval is None: continue
        lo, hi = interval
        updated = []
        for a, b in free:
            if hi <= a+eps or lo >= b-eps: updated.append((a,b)); continue
            if lo > a+eps: updated.append((a, min(b,lo)))
            if hi < b-eps: updated.append((max(a,hi),b))
        free = updated
    return tuple(free)


def width_profile(route, obstacles, nominal_width, *, floor_fn, height, epsilon):
    """Sample at every projected box corner and both sides of discontinuities.

    Within each straight route leg a box's cross-section changes at its corner
    projections. Midpoints cover each open interval; endpoints capture contacts.
    This profile records largest FREE interval, not the original wall spacing.
    """
    distances, start = {0.}, 0.
    for a, b in zip(route, route[1:]):
        length = math.dist(a,b); tangent = ((b[0]-a[0])/length,(b[1]-a[1])/length)
        distances.add(start+length)
        for box in obstacles:
            for x, y in box_corners_xy(box):
                s = (x-a[0])*tangent[0]+(y-a[1])*tangent[1]
                if 0 <= s <= length:
                    for offset in (-epsilon, 0., epsilon): distances.add(start+min(length,max(0.,s+offset)))
        start += length
    values = sorted(distances)
    values = sorted(set(values+[(a+b)/2 for a,b in zip(values,values[1:])]))
    class Route:
        route_polyline = route
        route_length = start
    result = []
    for s in values:
        point, tangent = route_point(Route, s)
        free = free_cross_section(point, tangent, nominal_width(s) if callable(nominal_width) else nominal_width, obstacles, floor_fn(point), height, epsilon)
        result.append((float(s), float(max((b-a for a,b in free), default=0.))))
    return tuple(result)
