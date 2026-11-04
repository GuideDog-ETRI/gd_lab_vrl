"""Backward gap stride task: the A' task (gast_gait.tasks, unchanged) + backward gap terms (terms.py).
Switches / weights: BACKSTRIDE_W_SPACING (-6), BACKSTRIDE_W_LIFT (-8), BACKSTRIDE_W_OVEREXT (-6),
BACKSTRIDE_W_PLACE (-4, A3), BACKSTRIDE_W_ORDER (-1), BACKSTRIDE_W_SUPPORT (-0.3), BACKSTRIDE_W_PRINT (-3) (A5); BACKSTRIDE_W_HIND_HEIGHT (0 = off; A6 -10), BACKSTRIDE_HIND_FLOOR (0.45); BACKSTRIDE_BACK_R17_SCALE (1; A8 4), BACKSTRIDE_PRINT_MODE (next | vacated (A8) | preslot (A9)), BACKSTRIDE_W_FRONT_FIRST (0; A9 -4), BACKSTRIDE_W_CROSS_FROM (0; A10 -10), BACKSTRIDE_W_ASIDE (0; A11 -6), BACKSTRIDE_W_NOSE_DOWN (0; A12 +0.5; A12 also BACKSTRIDE_SPACING_FLOOR 0.25), BACKSTRIDE_W_CLOSE_LAND (0; A13 -10), BACKSTRIDE_W_LEAD_REACH (0; A14 -10), BACKSTRIDE_BACK_GAP_MAX (0 = all; A14 0.20 m), BACKSTRIDE_W_STRIDE (0; A15 -6), BACKSTRIDE_W_CROSS_STEP (0; A16 -15), BACKSTRIDE_W_DIAG (0; A17 -3), BACKSTRIDE_W_LAND_MARGIN (0; A18 -20), BACKSTRIDE_BOTH_DIR (0; A18 1 = trot / stride / landing terms on forward gaps too and R17 off both ways);
0 = term off. BACKSTRIDE_BACK_R17=0 (A4): R17 off on backward gaps.
Local pilots 11-03 (from a1993): A = spacing + lift + R9 30 cm (placement as a1993, the front feet still land on
the braced hind feet in MuJoCo); B = GAIT_R19=0 only (placement unchanged); A3 = A + touchdown placement (placement
as 6566, but the hind foot a front foot will land on keeps bracing and the trunk sinks backward); A4 = A3 + no R17
on backward gaps (stopped: the user rejected the two-point moment it widens); A5 = A3 + the forward footstep
sequence mirrored (leader first, trailer lifts on three feet, trailer lands 0.30 m off the leader's print), R17 kept.
"""
from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from gd_lab.gast.teacher import GastRayIrRunnerCfg
from gd_lab.gast_backstride import terms
from gd_lab.gast_gait.tasks import FEET, FEET_BODIES, HIP_BODIES, GastGaitV3GapTeacherCfg
from gd_lab.gast_backstride.commands import BACK_GAP_MAX, NarrowBackGapVelocityCommand


def _w(name: str, default: float) -> float:
    return float(os.environ.get(f"BACKSTRIDE_W_{name}", str(default)))


W_SPACING = _w("SPACING", -6.0)   # /s per m below 0.40 m, per side (spacing 0 -> -2.4 /s per side)
W_LIFT = _w("LIFT", -8.0)         # /s per m below 8 cm over the slot, per front foot (R6's -8)
W_OVEREXT = _w("OVEREXT", -6.0)   # /s per m beyond 30 cm (R9's -6)
W_ORDER = _w("ORDER", -1.0)       # per front liftoff before its hind foot stepped (a1993: 17 % of front swings)
W_SUPPORT = _w("SUPPORT", -0.3)   # per front liftoff per foot short of three planted (A3: 76 % on two feet)
W_PRINT = _w("PRINT", -3.0)       # per front touchdown per m beyond 0.30 +- 0.10 m off the hind print (A3: 0.56)
W_HIND_HEIGHT = _w("HIND_HEIGHT", 0.0)  # /s per m a planted hind foot sits above 0.45 m below the trunk (A6)
W_FRONT_FIRST = _w("FRONT_FIRST", 0.0)  # /s a front foot over the slot while a hind foot is still before it (A9)
W_CROSS_FROM = _w("CROSS_FROM", 0.0)  # per front crossing per m its take-off is beyond 0.12 m off the hind pre-slot spot (A10)
W_NOSE_DOWN = _w("NOSE_DOWN", 0.0)  # bonus /s for 3-8 deg nose-down (rear raised) near a backward slot (A12)
W_CLOSE_LAND = _w("CLOSE_LAND", 0.0)  # per touchdown per m the same-side pair spacing is under 0.30 m (A13)
W_LEAD_REACH = _w("LEAD_REACH", 0.0)  # per hind crossing touchdown per m short of 0.32 m past its hip (A14)
W_STRIDE = _w("STRIDE", 0.0)  # per touchdown per m of step under 0.40 m near a backward slot (A15)
W_CROSS_STEP = _w("CROSS_STEP", 0.0)  # per crossing touchdown per m short of a 0.55 m step / 0.15 m past the far edge (A16)
W_DIAG = _w("DIAG", 0.0)  # /s per hind foot in the air with its diagonal front planted near a backward slot (A17)
W_LAND_MARGIN = _w("LAND_MARGIN", 0.0)  # per crossing touchdown per m under 0.10 m past the far edge (A18)
W_ASIDE = _w("ASIDE", 0.0)  # per front touchdown before the slot while its hind foot has crossed but not stepped again (A11)
W_PLACE = _w("PLACE", -4.0)       # per touchdown per m off the placement band (a1993: ~0.06 m x ~8 touchdowns/s)
SPACING_FLOOR = float(os.environ.get("BACKSTRIDE_SPACING_FLOOR", "0.40"))
HIND_FLOOR = float(os.environ.get("BACKSTRIDE_HIND_FLOOR", "0.45"))
LIFT_TARGET = float(os.environ.get("BACKSTRIDE_LIFT_TARGET", "0.08"))
BACK_R17 = os.environ.get("BACKSTRIDE_BACK_R17", "1") == "1"
BACK_R17_SCALE = float(os.environ.get("BACKSTRIDE_BACK_R17_SCALE", "1"))  # A8: R17 x4 on backward gaps
PRINT_MODE = os.environ.get("BACKSTRIDE_PRINT_MODE", "next")  # next: 0.30 m off the new hind print (A5); vacated: into the old one (A8)


@configclass
class GastGaitV3BackStrideTeacherCfg(GastGaitV3GapTeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        # always registered: it also hands the backward gap mask to the retention anchor. The reward manager skips
        # weight-0 terms, so "off" is a negligible weight (mask only).
        r.back_gap_spacing = RewardTermCfg(func=terms.stance_spacing_floor, weight=W_SPACING or -1e-9,
                                           params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                   "floor": SPACING_FLOOR})
        if W_LIFT:
            r.back_gap_front_lift = RewardTermCfg(func=terms.FrontLiftOverSlot, weight=W_LIFT,
                                                  params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                          "target": LIFT_TARGET})
        if W_PLACE:
            r.back_gap_placement = RewardTermCfg(func=terms.touchdown_placement, weight=W_PLACE,
                                                 params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                         "hip_cfg": HIP_BODIES, "front_max": 0.15, "hind_min": 0.18})
        seq = {"sensor_cfg": FEET, "asset_cfg": FEET_BODIES}
        if W_ORDER:
            r.back_gap_order = RewardTermCfg(func=terms.TrailerLiftoffOrder, weight=W_ORDER, params=dict(seq))
        if W_SUPPORT:
            r.back_gap_liftoff_support = RewardTermCfg(func=terms.TrailerLiftoffSupport, weight=W_SUPPORT,
                                                       params=dict(seq))
        if W_PRINT:
            if PRINT_MODE == "preslot":
                r.back_gap_footprint = RewardTermCfg(func=terms.TrailerIntoPreSlotSpot, weight=W_PRINT,
                                                     params=dict(seq, tol=0.08, reach=0.35))
            elif PRINT_MODE == "vacated":
                r.back_gap_footprint = RewardTermCfg(func=terms.TrailerIntoVacated, weight=W_PRINT,
                                                     params=dict(seq, tol=0.08))
            else:
                r.back_gap_footprint = RewardTermCfg(func=terms.TrailerFootprint, weight=W_PRINT,
                                                     params=dict(seq, target=0.30, tol=0.10))
        if W_HIND_HEIGHT:
            r.back_gap_hind_height = RewardTermCfg(func=terms.hind_stance_height_floor, weight=W_HIND_HEIGHT,
                                                   params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                           "floor": HIND_FLOOR})
        if W_FRONT_FIRST:
            r.back_gap_front_first = RewardTermCfg(func=terms.FrontCrossesFirst, weight=W_FRONT_FIRST,
                                                   params=dict(seq))
        if W_CROSS_FROM:
            r.back_gap_cross_from = RewardTermCfg(func=terms.FrontCrossFromPreSlotSpot, weight=W_CROSS_FROM,
                                                  params=dict(seq, tol=0.12, reach=0.35))
        if W_ASIDE:
            r.back_gap_hind_aside = RewardTermCfg(func=terms.HindStepsAsideFirst, weight=W_ASIDE, params=dict(seq))
        if W_LEAD_REACH:
            r.back_gap_lead_reach = RewardTermCfg(func=terms.LeaderCrossingReach, weight=W_LEAD_REACH,
                                                  params=dict(seq, hip_cfg=HIP_BODIES, target=0.32))
        if BACK_GAP_MAX > 0.0:
            self.commands.base_velocity.class_type = NarrowBackGapVelocityCommand
        if W_CROSS_STEP:
            r.back_gap_cross_step = RewardTermCfg(func=terms.BackGapCrossingStep, weight=W_CROSS_STEP,
                                                  params=dict(seq, floor=0.55, land_past=0.15))
        if W_DIAG:
            r.back_gap_diag_trot = RewardTermCfg(func=terms.HindSwingDiagonalPlanted, weight=W_DIAG, params=dict(seq))
        if W_LAND_MARGIN:
            r.gap_land_margin = RewardTermCfg(func=terms.CrossingLandingMargin, weight=W_LAND_MARGIN,
                                              params={**seq, "land_past": 0.10})
        if W_STRIDE:
            r.back_gap_stride = RewardTermCfg(func=terms.BackGapStrideFloor, weight=W_STRIDE,
                                              params=dict(seq, floor=0.40))
        if W_CLOSE_LAND:
            r.back_gap_close_land = RewardTermCfg(func=terms.touchdown_onto_pair_foot, weight=W_CLOSE_LAND,
                                                  params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                          "clear": 0.30})
        if W_NOSE_DOWN:
            r.back_gap_nose_down = RewardTermCfg(func=terms.BackGapNoseDown, weight=W_NOSE_DOWN,
                                                 params={"asset_cfg": FEET_BODIES})
        if terms.BOTH_DIR:
            r.gap_support = None  # A18: no three-point support on forward gaps either (user 11-05)
        elif not BACK_R17 and getattr(r, "gap_support", None) is not None:
            r.gap_support.func = terms.GapSupportForwardOnly  # A4: the hind foot may step back out of the way
        elif BACK_R17_SCALE != 1.0 and getattr(r, "gap_support", None) is not None:
            r.gap_support.func = terms.GapSupportBackBoost  # A8: one foot at a time over a backward slot
            r.gap_support.params["back_scale"] = BACK_R17_SCALE
        if W_OVEREXT:
            r.back_gap_overextension = RewardTermCfg(func=terms.back_gap_overextension, weight=W_OVEREXT,
                                                     params={"sensor_cfg": FEET, "asset_cfg": FEET_BODIES,
                                                             "hip_cfg": HIP_BODIES, "limit": 0.30})


@configclass
class GastGaitV3BackStrideRunnerCfg(GastRayIrRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/gait_v3_backstride"
        self.algorithm.class_name = "gd_lab.gast_backstride.anchor:BackStrideAnchorPPO"


TASK = "GastGaitV3BackStride"
TASK_ID = "Gd-GastGaitV3BackStride-Rbq10-Dreamwaq-v0"
