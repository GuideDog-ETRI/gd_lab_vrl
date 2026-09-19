"""On-policy runner with importlib class resolution and full resume state."""

from __future__ import annotations

import importlib

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .actor_critic import DreamwaqActorCritic


def _resolve_class(name: str) -> type:
    """Resolve ``"<module>:<attr>"``; bare names fall back to the rsl_rl namespaces."""
    if ":" in name:
        module_name, attr = name.split(":")
        return getattr(importlib.import_module(module_name), attr)
    for module_name in ("rsl_rl.modules", "rsl_rl.algorithms"):
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"Cannot resolve class '{name}'; use the '<module>:<attr>' form")


class DreamwaqRunner(OnPolicyRunner):
    """Upstream OnPolicyRunner with three additions: policy/algorithm classes are
    resolved via importlib (so project classes are reachable from config strings),
    the adaptive learning rate and CENet optimizer survive save/load, and resume
    continues at the next iteration instead of re-running the saved one."""

    # Mirrors the upstream ``_construct_algorithm`` flow; only the two ``eval``
    # class lookups are replaced by ``_resolve_class``.
    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        policy_class = _resolve_class(self.policy_cfg.pop("class_name"))
        policy = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)

        alg_class = _resolve_class(self.alg_cfg.pop("class_name"))
        alg = alg_class(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        alg.init_storage("rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions])
        return alg

    #: Deployment contract, captured from the live environment by the entry
    #: script (this layer is env-agnostic and cannot reach ``gd_lab.deploy``).
    #: Carried in every checkpoint because export runs without a simulator.
    deploy_context: dict | None = None

    def save(self, path: str, infos: dict | None = None) -> None:
        extra = {"learning_rate": self.alg.learning_rate}
        policy = self.alg.policy
        if isinstance(policy, DreamwaqActorCritic):
            extra["cenet_optimizer_state_dict"] = policy.cenet.optimizer.state_dict()
        if self.deploy_context is not None:
            extra["deploy_context"] = self.deploy_context
        infos = {**(infos or {}), "gd_lab": extra}
        super().save(path, infos)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        extra = (infos or {}).get("gd_lab", {})
        if "learning_rate" in extra:
            self.alg.learning_rate = extra["learning_rate"]
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate
        policy = self.alg.policy
        if load_optimizer and isinstance(policy, DreamwaqActorCritic) and "cenet_optimizer_state_dict" in extra:
            policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
        # The checkpoint was written after finishing ``iter``; continue from the next one.
        self.current_learning_iteration += 1
        return infos

    def export_inference_state(self) -> dict[str, torch.Tensor]:
        self.eval_mode()
        return self.alg.policy.state_dict()
