"""Narrow-passage tile geometry, shared by the mesh generator and the torch rewards/commands.

Coordinates are tile-local with the env origin at the tile centre (x forward = the passage direction at spawn).
Every wall is an axis-aligned box: (x0, x1, y0, y1), floor at z = 0, height WALL_H.

Kinds (width w from the curriculum difficulty, W_MAX at level 0 -> W_MIN at the hardest level):
- corridor : straight corridor x in [-1.5, 2.5], walls at |y| = w/2.
- door     : two transverse walls (x in [0, 0.2] and [1.6, 1.8]) with openings of width w; the second opening is
             shifted laterally by ``sign`` * DOOR_SHIFT (gate -> gate offset as at ticket gates / train doors).
- bend     : corridor along +x into a 90 deg corner at x = 1.0, leaving along ``sign`` * y.
- dead_end : (v2) the corridor closed by a cross wall at x = dead_end_x(difficulty); never passable.
The path is the passage centre line (what gdm's local path would follow); success = past the exit line.
``block_s`` (v2) = path arclength of the line the robot front must not cross when the passage is blocked for it:
the start of the narrow section (corridor/bend walls, first door), or the dead-end wall minus FRONT_MARGIN.
"""
from __future__ import annotations

import math

W_MAX, W_MIN = 1.20, 0.60
WALL_T, WALL_H = 0.10, 1.00
DOOR_SHIFT = 0.25
SPAWN_X = -2.2
KINDS = {"corridor": 1, "door": 2, "bend": 3, "dead_end": 4}
PATH_START = -3.5
FRONT_MARGIN = 0.10
MAX_RECTS, MAX_PATH = 6, 4
FAR = 1.0e3


def width_from_difficulty(difficulty: float, w_max: float = W_MAX, w_min: float = W_MIN) -> float:
    d = min(1.0, max(0.0, float(difficulty)))
    return w_max - (w_max - w_min) * d


def dead_end_x(difficulty: float) -> float:
    """Dead-end wall position: 1.5 m into the corridor (easy, seen late) .. at its mouth -1.2 m (hard)."""
    d = min(1.0, max(0.0, float(difficulty)))
    return 1.5 - 2.7 * d


def build(kind: str, width: float, sign: int, end_x: float = 0.0) -> dict:
    """{'rects': [(x0, x1, y0, y1)], 'path': [(x, y)], 'exit': (axis, direction, value)}."""
    w, t = float(width), WALL_T
    h = w / 2
    if kind == "corridor":
        rects = [(-1.5, 2.5, h, h + t), (-1.5, 2.5, -h - t, -h)]
        path = [(-3.5, 0.0), (3.5, 0.0)]
        exit_ = (0, 1.0, 2.8)
    elif kind == "door":
        c = sign * DOOR_SHIFT
        rects = [(0.0, 0.2, h, 3.5), (0.0, 0.2, -3.5, -h),
                 (1.6, 1.8, c + h, 3.5), (1.6, 1.8, -3.5, c - h)]
        path = [(-3.5, 0.0), (0.1, 0.0), (1.7, c), (3.5, c)]
        exit_ = (0, 1.0, 2.3)
    elif kind == "bend":
        cx = 1.0
        s = float(sign)
        # Outer wall on the far side of the turn, inner wall on the turn side (before mirroring, sign=+1 = left).
        rects = [(-1.5, cx + h + t, -h - t, -h),          # outer, along x
                 (-1.5, cx - h, h, h + t),                # inner, along x
                 (cx + h, cx + h + t, -h - t, 3.0),       # outer, along y
                 (cx - h - t, cx - h, h, 3.0)]            # inner, along y
        if s < 0:
            rects = [(x0, x1, -y1, -y0) for x0, x1, y0, y1 in rects]
        path = [(-3.5, 0.0), (cx, 0.0), (cx, 3.5 * s)]
        exit_ = (1, s, 2.6)
    elif kind == "dead_end":
        e = float(end_x)
        rects = [(-1.5, e + t, h, h + t), (-1.5, e + t, -h - t, -h), (e, e + t, -h, h)]
        path = [(-3.5, 0.0), (3.5, 0.0)]
        exit_ = (0, 1.0, FAR)
    else:
        raise ValueError(f"unknown narrow kind {kind!r}")
    entry = {"corridor": -1.5, "door": 0.0, "bend": -1.5}.get(kind)
    block_s = (entry if entry is not None else float(end_x) - FRONT_MARGIN) - PATH_START
    return {"kind": kind, "width": w, "sign": int(sign), "rects": rects, "path": path, "exit": exit_,
            "block_s": block_s}


def padded(geometry: dict | None) -> tuple[list, list, tuple, int]:
    """Fixed-size rows for the torch tables: rects padded far away, path padded with its last point."""
    if geometry is None:
        rects = [(FAR, FAR + 1, FAR, FAR + 1)] * MAX_RECTS
        return rects, [(0.0, 0.0)] * MAX_PATH, (0, 1.0, FAR), 0
    rects = list(geometry["rects"]) + [(FAR, FAR + 1, FAR, FAR + 1)] * (MAX_RECTS - len(geometry["rects"]))
    path = list(geometry["path"]) + [geometry["path"][-1]] * (MAX_PATH - len(geometry["path"]))
    return rects, path, geometry["exit"], KINDS[geometry["kind"]]


def path_length(path) -> float:
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
