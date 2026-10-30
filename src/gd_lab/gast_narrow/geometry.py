"""Narrow-passage tile geometry, shared by the mesh generator and the torch rewards/commands.

Coordinates are tile-local with the env origin at the tile centre (x forward = the passage direction at spawn).
Every wall is an axis-aligned box: (x0, x1, y0, y1), floor at z = 0, height WALL_H.

Kinds (width w from the curriculum difficulty, W_MAX at level 0 -> W_MIN at the hardest level):
- corridor : straight corridor x in [-1.5, 2.5], walls at |y| = w/2.
- door     : two transverse walls (x in [0, 0.2] and [1.6, 1.8]) with openings of width w; the second opening is
             shifted laterally by ``sign`` * DOOR_SHIFT (gate -> gate offset as at ticket gates / train doors).
- bend     : corridor along +x into a 90 deg corner at x = 1.0, leaving along ``sign`` * y.
- dead_end : (v2) the corridor closed by a cross wall at x = dead_end_x(difficulty); never passable.
- turn_*   : (v2.4) an in-place turn pocket around the tile centre; ``width`` is the free radius r (centre -> the
             nearest obstacle). aisle: two bench fronts (0.45 m high) at |y| = r, open ends; vestibule: back wall at
             y = -r, side walls at |x| = r + 0.15, a pole at distance r on the open side (``sign`` = its side);
             room: walls at |y| = r and |x| = 1.3 r (``sign`` < 0: the long axis along y). Kind "turn" for all three.
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
KINDS = {"corridor": 1, "door": 2, "bend": 3, "dead_end": 4, "turn": 5}
TURN_SHAPES = ("turn_aisle", "turn_vestibule", "turn_room")
BENCH_H, POLE_W, ROOM_ASPECT, VESTIBULE_SIDE = 0.45, 0.04, 1.3, 0.15
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


def build_turn(shape: str, radius: float, sign: int) -> dict:
    """Turn pocket: rects (and their heights) whose nearest point to the tile centre is ``radius`` away."""
    r, t, s = float(radius), WALL_T, float(sign)
    if shape == "turn_aisle":
        rects = [(-3.0, 3.0, r, r + 0.45), (-3.0, 3.0, -r - 0.45, -r)]
        heights = [BENCH_H, BENCH_H]
    elif shape == "turn_vestibule":
        side = r + VESTIBULE_SIDE
        c = r / math.sqrt(2.0) + POLE_W / 2  # pole centre on the diagonal, its near corner at distance r
        rects = [(-side - t, side + t, -r - t, -r), (side, side + t, -r, 2.0), (-side - t, -side, -r, 2.0),
                 (s * c - POLE_W / 2, s * c + POLE_W / 2, c - POLE_W / 2, c + POLE_W / 2)]
        heights = [WALL_H, WALL_H, WALL_H, WALL_H]
    elif shape == "turn_room":
        a, b = ROOM_ASPECT * r, r
        if s < 0:
            a, b = b, a
        rects = [(-a - t, a + t, b, b + t), (-a - t, a + t, -b - t, -b), (a, a + t, -b, b), (-a - t, -a, -b, b)]
        heights = [WALL_H] * 4
    else:
        raise ValueError(f"unknown turn shape {shape!r}")
    return {"kind": "turn", "shape": shape, "width": r, "sign": int(sign), "rects": rects, "heights": heights,
            "path": [(-0.5, 0.0), (0.5, 0.0)], "exit": (0, 1.0, FAR), "block_s": 0.0}


def build(kind: str, width: float, sign: int, end_x: float = 0.0) -> dict:
    """{'rects': [(x0, x1, y0, y1)], 'path': [(x, y)], 'exit': (axis, direction, value)}."""
    if kind in TURN_SHAPES:
        return build_turn(kind, width, sign)
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
