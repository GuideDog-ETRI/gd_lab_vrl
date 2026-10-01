"""Blind-start VRL experiments (stage 2-A / 2-B), kept outside ``gd_lab``.

Importing registers three tasks through gd_lab's registry:
  Gd-VrlBlindStart-Rbq10-Dreamwaq-v0         2-A, camera-free
  Gd-VrlBlindStart-Rbq10-Dreamwaq-Vision-v0  2-B, camera-visible
  Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0  2-R, camera-free raycast visibility
"""

from gd_lab.core import registry

_AGENT = "gd_lab.teachers.bivt.agent_cfg:BlindStartRunnerCfg"

BLIND_START_TASK = registry.register_task(
    task="VrlBlindStart", robot="Rbq10", method="Dreamwaq",
    env_cfg="gd_lab.teachers.bivt.tasks:BlindStartEnvCfg", agent_cfg=_AGENT,
)
BLIND_START_CAMERA_TASK = registry.register_task(
    task="VrlBlindStart", robot="Rbq10", method="Dreamwaq", mode="Vision",
    env_cfg="gd_lab.teachers.bivt.tasks:BlindStartCameraEnvCfg", agent_cfg=_AGENT,
)
BLIND_START_RAYCAST_TASK = registry.register_task(
    task="VrlBlindStartRaycast", robot="Rbq10", method="Dreamwaq",
    env_cfg="gd_lab.teachers.bivt.tasks:BlindStartRaycastEnvCfg", agent_cfg=_AGENT,
)
