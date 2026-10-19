"""GAST-Nav v1 shared schema: tensor shapes, codes and module ownership.

Design: docs/design/GAST_NAV_V1_DESIGN.md (v0.3.1, appendix C). All numeric thresholds live in
configs/gast_nav/*.yaml; this file only fixes layouts both the map side (C2) and the policy side (C1) use.
"""

# Wide obstacle map (3.2a): body-yaw frame, cell centres, x-fastest.
WIDE_NX, WIDE_NY, WIDE_RES = 49, 43, 0.10          # x -2.4..+2.4, y -2.1..+2.1
WIDE_X0, WIDE_Y0 = -2.4, -2.1
BANDS = (("B1", 0.03, 0.30), ("B2", 0.30, 0.80), ("B3", 0.80, 2.20))   # metres above estimated support
SUBBIN = 0.10                                       # C2.3 free rule: every 10 cm sub-bin traversed
WIDE_FRAMES = 3                                     # current + 2 past captures (10 Hz)

# Cell state codes (2 bit), C2.5. 3 is reserved: a packet containing it is rejected.
UNKNOWN, FREE, OCCUPIED, RESERVED = 0, 1, 2, 3

# Age: 4 bit, 0.1 s steps rounded up; 15 = >=1.5 s or never observed. TTL 0.5 s -> UNKNOWN.
AGE_STEP_S, AGE_MAX_CODE = 0.1, 15

# Support height (C2.2): uint8, base-relative, 2 cm steps from -1.0 m; 255 = unknown.
SUPPORT_Z0, SUPPORT_STEP, SUPPORT_UNKNOWN = -1.0, 0.02, 255

# Near clearance map (3.2b, C2.4): 5 cm, x -0.9..+0.9, y -0.6..+0.6.
NEAR_NX, NEAR_NY, NEAR_RES = 37, 25, 0.05
NEAR_X0, NEAR_Y0 = -0.9, -0.6
NEAR_CLEAR_MAX = 0.5                                # metres, uint8 = round(d / 0.5 * 255)

# Network input encoding (decoded on the policy side; the map side must produce exactly this).
#   wide: per frame per band [occupied, free, age/1.5]  -> WIDE_FRAMES * 3 * 3 = 27 channels of (WIDE_NY, WIDE_NX)
#   near: [low_clear/0.5, high_clear/0.5, unknown]       -> 3 channels of (NEAR_NY, NEAR_NX), current capture only
WIDE_CHANNELS = WIDE_FRAMES * len(BANDS) * 3
NEAR_CHANNELS = 3

# Command / phase vocabulary (3.4, 3.5, C3).
MODES = ("LEGACY", "NAV", "TURN")
PHASES = ("APPROACH", "PRE_BRAKE", "LANDING_ENTERED", "DWELL", "RELEASED", "DESCEND")
SEGMENT_TYPES = ("FLAT", "PASSAGE", "STAIR_UP", "LANDING", "STAIR_DOWN", "TURN_SPOT")
RELEASE_SOURCES = ("AUTO", "GDM", "USER")
CONTROLLERS = ("NAV_POLICY", "HOLD_3906", "DAMP")
HEALTH = ("OK", "TERRAIN_STALE", "POSE_STALE", "SUPERVISOR_RISK", "INFER_FAIL", "NAV1_INVALID")
CONDITION_KEYS = ("y_off", "margin", "turn_cap")    # v_cap lives in NAV1 only

# Module ownership for UPDATE159 (to avoid edit conflicts between the two Codex sessions):
#   map side (C2, Codex90):   support.py, obstacle_map.py, near_clearance.py, warp.py, packets.py (OBS1)
#   policy side (C1/C3/C4, Codex5090): migration.py, actor_critic.py, trainable.py, ppo.py,
#                             commands.py (NAV1/ACK), landing_fsm.py, supervisor.py
#   shared: this file (change only by agreement through Claude).

# UPDATE179 / D-5: independent observation, old nav_features remains 16-wide.
ROUTE_OBS_KEY = "nav_route_features"
ROUTE_FEATURE_KEYS = ("route_yaw_sin", "route_yaw_cos", "route_lateral_error_m")
ROUTE_FEATURE_DIM = len(ROUTE_FEATURE_KEYS)
ROUTE_FEATURE_VERSION = 1
# route_axis_yaw in NAV1 is world/odom yaw. Encode sin/cos(axis - body yaw).
# Lateral = (robot - closest path point) dot directed path left normal, metres.
# It is geometric crosstrack, NOT crosstrack minus the separately supplied c.y_off.
