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
