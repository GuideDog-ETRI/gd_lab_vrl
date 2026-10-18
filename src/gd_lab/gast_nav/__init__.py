"""GAST-Nav v1: narrow-passage, clockwise in-place turn and stair-alignment teacher on top of Ray+IR 3906.

Design: docs/design/GAST_NAV_V1_DESIGN.md. Shared layouts: gd_lab.gast_nav.schema.
"""

# Local registration: existing global registry stays untouched.
import gymnasium as _gym
_TASK_ID = 'Gd-GastNavV1-Rbq10-Dreamwaq-v0'
if _TASK_ID not in _gym.registry:
    _gym.register(id=_TASK_ID,entry_point='gd_lab.gast_nav.env:GastNavEnv',disable_env_checker=True,
                  kwargs={'env_cfg_entry_point':'gd_lab.gast_nav.env_cfg:GastNavEnvCfg'})
