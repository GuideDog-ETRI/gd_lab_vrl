"""Gait quality v3 task: the 6566 recipe (backward gap V2) + the v3 posture / touchdown / leg-following terms.

Design: docs/design/GAIT_V3_DESIGN.md. Start: 3906 (user decision 11-01). The 3906 and backgap code stay unchanged.
Each term can be switched off for ablations with GAIT_R<n>=0 (R1, R4..R10, R15..R18); GAIT_R11=1 re-enables
lead_foot_stall; GAIT_R13=1 keeps backward_nose_up. G3: R1, R4, R5, R7, R9, R10 are off on stair and slope tiles.
R1 allows any pitch between level and the terrain rise under the trunk (GAIT_R1_BAND=0: the G1 version, pitch minus
rise; GAIT_R1_GROUND=0: the G0 version, absolute pitch).
"""
from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from gd_lab.gast_backgap.tasks import GastRayIrV21Gap50BackV2TeacherCfg
from gd_lab.gast_gait import terms

FEET = SceneEntityCfg("contact_forces", body_names=".*_foot")
FEET_BODIES = SceneEntityCfg("robot", body_names=".*_foot")
HIP_BODIES = SceneEntityCfg("robot", body_names=".*_hip")

# weights (design section 3; event terms are per touchdown, rates per second)
W_NOSE_UP = -4.0           # R1 /s per rad over 3 deg (8 deg -> -0.35 /s, 15 deg -> -0.84 /s)
W_TOUCHDOWN = -0.3         # R4 per touchdown per m/s over 0.3 (1.2 m/s -> -0.27); -0.08 was -0.011 /s in G0 (inert)
W_IMPACT = -0.1            # R5 per touchdown per unit of (peak / static share) over 2.0 (2.6x -> -0.06)
W_CLEARANCE = -8.0         # R6 /s per m of shortfall below 7 cm mid-swing (3906: 4.6 cm -> about -0.1 /s mean)
W_SWING_FLOOR = -1.0       # R7 per touchdown per s under 0.14 s (0.10 s -> -0.04); -2 cost 3906 -0.3 /s walking backward
W_SLIP = -0.3              # R8
W_OVEREXT = -6.0           # R9 /s per m beyond 18 cm (22 cm on gaps, x1.5): 30 cm behind -> -0.72 /s on a gap
W_SWING_SPEED = -1.0       # R10 /s per m/s over 3.0 (3.5 on gaps; 3906 gap swings peak 3.5-4.5 m/s)
W_LEAD_STALL = -2.0        # R11 (off by default since G3: the brace is support, R17 prices its absence instead)
W_GAP_SUPPORT = -0.5       # R17 /s per foot short of 3 in contact near a slot (a trot over the slot: about -0.5 /s)
GAP_FOOTHOLD_MARGIN = 0.06  # 6566: 0.04; backward hind feet landed on the slot edge (G1 / G2 focus replays 11-01)
W_GAP_FOOTHOLD = -0.4      # 6566: -0.25
W_LIFT_CAP = -8.0          # R15 /s per m of swing sole above 9 cm (level ground only); 10 cm -> -0.08 /s per foot
W_BACK_OVERSPEED = -1.0    # R16 /s per m/s of backward speed above the command + 0.2 on gap / stair tiles


OFF_BY_DEFAULT = {11}


def _on(n: int) -> bool:
    return os.environ.get(f"GAIT_R{n}", "0" if n in OFF_BY_DEFAULT else "1") == "1"


SHAPED_OFF = {"off_shaped": True}  # G3: gait quality terms are off on stair and slope tiles


@configclass
class GastGaitV3GapTeacherCfg(GastRayIrV21Gap50BackV2TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        if _on(1):
            r.trunk_nose_up = RewardTermCfg(func=terms.TrunkNoseUp, weight=W_NOSE_UP,
                                            params={"asset_cfg": FEET_BODIES, "limit_deg": 3.0, "down_deg": 6.0,
                                                    "gap_limit_deg": 12.0, "gap_down_deg": 10.0,
                                                    "ground_relative": os.environ.get("GAIT_R1_GROUND", "1") == "1",
                                                    "level_band": os.environ.get("GAIT_R1_BAND", "1") == "1",
                                                    **SHAPED_OFF})
        if _on(4):
            r.feet_touchdown = RewardTermCfg(func=terms.feet_touchdown_v3, weight=W_TOUCHDOWN,
                                             params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                     "free_speed": 0.3, "stair_free_speed": 0.6, **SHAPED_OFF})
        if _on(5):
            r.feet_impact_force = RewardTermCfg(func=terms.FeetImpactForce, weight=W_IMPACT,
                                                params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                        "free_ratio": 2.0, "stair_free_ratio": 2.5, **SHAPED_OFF})
        if _on(6):
            r.foot_clearance = RewardTermCfg(func=terms.foot_clearance_ground, weight=W_CLEARANCE,
                                             params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES, "target": 0.07})
        if _on(7):
            r.swing_time_floor = RewardTermCfg(func=terms.swing_time_floor, weight=W_SWING_FLOOR,
                                               params={"sensor_cfg": FEET, "floor_s": 0.14, **SHAPED_OFF})
        if _on(8):
            r.feet_slipping.weight = W_SLIP
        if _on(9):
            r.stance_overextension = RewardTermCfg(func=terms.stance_overextension, weight=W_OVEREXT,
                                                   params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                           "hip_cfg": HIP_BODIES, "limit": 0.18,
                                                           "gap_limit": 0.22, "gap_scale": 1.5, **SHAPED_OFF})
        if _on(10):
            r.swing_speed_cap = RewardTermCfg(func=terms.swing_speed_cap, weight=W_SWING_SPEED,
                                              params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES, "cap": 3.0,
                                                      "gap_cap": 3.5, **SHAPED_OFF})
        if _on(11):
            r.lead_foot_stall = RewardTermCfg(func=terms.lead_foot_stall, weight=W_LEAD_STALL,
                                              params={"sensor_cfg": FEET, "stall_s": 0.8, "cap_s": 1.0,
                                                      "min_trunk_speed": 0.15})
        if _on(15):
            r.foot_lift_cap = RewardTermCfg(func=terms.FootLiftCap, weight=W_LIFT_CAP,
                                            params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES, "cap": 0.09})
        if _on(16):
            r.backward_overspeed = RewardTermCfg(func=terms.backward_overspeed, weight=W_BACK_OVERSPEED, params={})
        if _on(17):
            r.gap_support = RewardTermCfg(func=terms.GapSupport, weight=W_GAP_SUPPORT,
                                          params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES, "min_feet": 3})
        if _on(18):
            r.gap_foothold_margin.params["margin"] = GAP_FOOTHOLD_MARGIN
            r.gap_foothold_margin.weight = W_GAP_FOOTHOLD
        if os.environ.get("GAIT_R13", "0") != "1":
            r.backward_nose_up = None  # absorbed by R1


TASK = "GastGaitV3Gap"
TASK_ID = "Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0"
