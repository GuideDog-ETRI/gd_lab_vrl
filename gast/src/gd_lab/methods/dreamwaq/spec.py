"""Observation contract of the blind DreamWaQ method - the single source of truth.

Everything layout-dependent derives from this module: the IsaacLab observation
groups, the CENet input/latest-step slicing, the actor input width, the critic
velocity-target slice, the mirror permutations, and the export dims. The
``height_scan`` width is resolved at runtime from the scanner grid.
"""

from __future__ import annotations

from gd_lab.core.types import ObsSpec, ObsSpecSet, ObsTermSpec

NUM_JOINTS = 12
HISTORY_LENGTH = 5

# The proprio block is shared by the policy (noisy, stacked) and critic (clean,
# current) groups; its order is a deploy contract - the on-robot observation
# packer fills exactly this layout, and ``payload`` extends it by one.
_PROPRIO_TERMS = (
    ObsTermSpec("base_ang_vel", 3, "ang_vel"),
    ObsTermSpec("projected_gravity", 3, "lin_vec"),
    ObsTermSpec("velocity_commands", 3, "commands"),
    ObsTermSpec("joint_pos", NUM_JOINTS, "joint"),
    ObsTermSpec("joint_vel", NUM_JOINTS, "joint"),
    ObsTermSpec("actions", NUM_JOINTS, "joint"),
)

# The operator-known trunk payload. Declared AFTER the proprio block in both
# groups so toggling it leaves that deploy-stable prefix untouched.
_PAYLOAD_TERM = ObsTermSpec("payload", 1, "identity")

DREAMWAQ_SPEC = ObsSpecSet(
    policy=ObsSpec(terms=(*_PROPRIO_TERMS, _PAYLOAD_TERM), history=HISTORY_LENGTH),
    critic=ObsSpec(
        terms=(
            *_PROPRIO_TERMS,
            ObsTermSpec("base_lin_vel", 3, "lin_vec"),
            ObsTermSpec("feet_around_height_from_terrain", 16, "feet16"),
            ObsTermSpec("feet_contact", 4, "feet4"),
            ObsTermSpec("feet_contact_forces", 12, "feet_vec3"),
            ObsTermSpec("friction_coeff", 2, "identity"),
            ObsTermSpec("base_mass_offset", 1, "identity"),
            ObsTermSpec("actuator_gain_scale", 2 * NUM_JOINTS, "gain24"),
            ObsTermSpec("push_delta_v", 3, "lin_vec"),
            ObsTermSpec("height_scan", None, "scan"),
            _PAYLOAD_TERM,
        ),
        history=1,
    ),
)

ONE_STEP_OBS_DIM = DREAMWAQ_SPEC.policy.per_step
POLICY_OBS_DIM = DREAMWAQ_SPEC.policy.total

# (start, stop) into the raw critic group; usable without resolving height_scan
# because base_lin_vel precedes it.
VELOCITY_TARGET_SLICE = (
    DREAMWAQ_SPEC.critic.offset("base_lin_vel"),
    DREAMWAQ_SPEC.critic.offset("base_lin_vel") + 3,
)
