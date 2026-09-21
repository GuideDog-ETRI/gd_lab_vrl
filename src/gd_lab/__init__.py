"""gd_lab: RBQ10 quadruped locomotion training framework.

Importing this package registers all task combinations; environment and agent
configs are referenced by entry-point strings, so no simulator import happens here.
"""

from gd_lab.core import registry

_ENV = "gd_lab.tasks.blind_rough:BlindRoughEnvCfg"
_AGENT = "gd_lab.agents.dreamwaq_ppo_cfg:DreamwaqRunnerCfg"

registry.register_task(task="Blind", robot="Rbq10", method="Dreamwaq", env_cfg=_ENV, agent_cfg=_AGENT)
registry.register_task(
    task="Blind", robot="Rbq10", method="Dreamwaq", mode="Play",
    env_cfg="gd_lab.tasks.blind_rough:BlindRoughEnvCfg_PLAY", agent_cfg=_AGENT,
)
registry.register_task(
    task="Blind", robot="Rbq10", method="Dreamwaq", mode="Gamepad",
    env_cfg="gd_lab.tasks.blind_rough:BlindRoughEnvCfg_GAMEPAD", agent_cfg=_AGENT,
)

# Vision-RL. Same module name as the blind package on purpose - the two live in
# separate trees (gd_lab / gd_lab_vrl) and are never loaded together, so only the
# task IDs need to differ. Run this tree with
# ``PYTHONPATH=<vrl-repo>/src`` rather than installing it,
# which would repoint the shared editable install and hijack blind runs.
#
# Two families, because the two stages need different scenes:
#   Vrl / Vrl-Play              stage 1 teacher. Camera-FREE -- the teacher reads
#                               the privileged height_scan, so rendering four
#                               belly cameras through PPO would be pure waste.
#   Vrl-Vision / Vrl-VisionPlay stage 3 distillation and student validation.
#                               Same observation contract, cameras added.
_VRL_AGENT = "gd_lab.agents.dreamwaq_vrl_ppo_cfg:DreamwaqVrlRunnerCfg"

registry.register_task(
    task="Vrl", robot="Rbq10", method="Dreamwaq",
    env_cfg="gd_lab.tasks.vrl_teacher:VrlTeacherEnvCfg", agent_cfg=_VRL_AGENT,
)
registry.register_task(
    task="Vrl", robot="Rbq10", method="Dreamwaq", mode="Play",
    env_cfg="gd_lab.tasks.vrl_teacher:VrlTeacherEnvCfg_PLAY", agent_cfg=_VRL_AGENT,
)
registry.register_task(
    task="Vrl", robot="Rbq10", method="Dreamwaq", mode="Vision",
    env_cfg="gd_lab.tasks.vrl_rough:VisionRoughEnvCfg", agent_cfg=_VRL_AGENT,
)
registry.register_task(
    task="Vrl", robot="Rbq10", method="Dreamwaq", mode="VisionPlay",
    env_cfg="gd_lab.tasks.vrl_rough:VisionRoughEnvCfg_PLAY", agent_cfg=_VRL_AGENT,
)
