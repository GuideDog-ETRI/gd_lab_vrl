"""Blind-start VRL experiments (stage 2-A / 2-B), kept outside ``gd_lab``.

Importing registers these tasks through gd_lab's registry:
  Gd-VrlBlindStart-Rbq10-Dreamwaq-v0         2-A, camera-free
  Gd-VrlBlindStart-Rbq10-Dreamwaq-Vision-v0  2-B, camera-visible
  Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0  2-R, camera-free raycast visibility
  Gd-VrlGapFinetune{Baseline,Intrusion,Clean}Raycast-Rbq10-Dreamwaq-v0  opt-in gap fine-tuning arms
"""

from gd_lab.core import registry

from .gap_guard import GAP_ARMS, V2_ARMS

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

GAP_FINETUNE_TASKS = tuple(
    registry.register_task(
        task=f"VrlGapFinetune{arm}Raycast", robot="Rbq10", method="Dreamwaq",
        env_cfg=f"gd_lab.teachers.bivt.tasks:GapFinetune{arm}RaycastEnvCfg", agent_cfg=_AGENT,
    )
    for arm in GAP_ARMS
)

GAP_V2_TASKS = tuple(
    registry.register_task(
        task=f"VrlGapFinetune{arm}Raycast", robot="Rbq10", method="Dreamwaq",
        env_cfg=f"gd_lab.teachers.bivt.tasks:GapFinetune{arm}RaycastEnvCfg", agent_cfg=_AGENT,
    )
    for arm in V2_ARMS
)
