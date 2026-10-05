"""On-policy runner with importlib class resolution and full resume state."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from rsl_rl.algorithms import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .actor_critic import DreamwaqActorCritic
from .online_rollout import install_episode_collector
from .online_top5 import DEFAULT_TOP5_CRITERIA, Top5CriteriaReloader, rank_and_save_top5
from .terrain_resume import (
    capture_family_level_means,
    read_family_level_means_from_tensorboard,
    restore_family_start_levels,
)


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

    def configure_online_top5(
        self, family_columns: dict[str, list[int]], spacing: int = 100,
        min_platform_gap_mean_level: float = 8.0,
    ) -> None:
        """Enable rollout-only ranking for Vision runs; blind runs stay unchanged."""
        self._top5_records = install_episode_collector(self.env, family_columns)
        self._top5_family_columns = {name: list(columns) for name, columns in family_columns.items()}
        self._top5_spacing = spacing
        self._top5_min_platform_gap_mean_level = min_platform_gap_mean_level

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if not hasattr(self, "_top5_records"):
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        defaults = dict(DEFAULT_TOP5_CRITERIA)
        defaults["top5_min_spacing"] = self._top5_spacing
        defaults["min_platform_gap_mean_level"] = self._top5_min_platform_gap_mean_level
        criteria_file = os.environ.get("GD_LAB_TOP5_CRITERIA_FILE")
        if criteria_file is None:
            criteria_file = Path(__file__).resolve().parents[3] / "configs/online_top5.json"
        self._top5_criteria_reloader = Top5CriteriaReloader(criteria_file, defaults)

        original_update = self.alg.update
        next_iteration = self.current_learning_iteration
        class SavedStop(Exception):
            pass

        def update_and_rank():
            nonlocal next_iteration
            result = original_update()
            self.current_learning_iteration = next_iteration
            records = self._top5_records
            local = list(records)
            records.clear()
            if self.is_distributed:
                gathered = [None] * self.gpu_world_size
                dist.all_gather_object(gathered, local)
                rank = dist.get_rank()
                episodes = [row for batch in gathered for row in batch] if rank == 0 else []
            else:
                rank = 0
                episodes = local
            status = {"selected": False, "saved": False, "error": None}
            if rank == 0 and not self.disable_logs and self.log_dir is not None:
                criteria, criteria_message = self._top5_criteria_reloader.refresh()
                if criteria_message:
                    level = "INFO" if criteria_message.startswith("Reloaded") else "WARN"
                    print(f"[{level}] {criteria_message}", flush=True)
                status = rank_and_save_top5(
                    rank, episodes, Path(self.log_dir) / "best_top5", next_iteration,
                    criteria["top5_min_spacing"], self.save, criteria=criteria,
                )
            if self.is_distributed:
                message = [status]
                dist.broadcast_object_list(message, src=0)
                status = message[0]
            if status["error"]:
                raise RuntimeError(status["error"])
            stop_file = os.environ.get('GD_LAB_STOP_FILE')
            requested = bool(stop_file and Path(stop_file).exists()) if rank == 0 else False
            if self.is_distributed:
                message = [requested]
                dist.broadcast_object_list(message, src=0)
                requested = message[0]
            if requested:
                if rank == 0:
                    self.save(os.path.join(self.log_dir, f'model_{next_iteration}.pt'))
                    print(f'[SAVED_STOP] iteration={next_iteration}', flush=True)
                if self.is_distributed:
                    dist.barrier()
                raise SavedStop()
            next_iteration += 1
            return result

        self.alg.update = update_and_rank
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        except SavedStop:
            return
        finally:
            self.alg.update = original_update

    def save(self, path: str, infos: dict | None = None) -> None:
        infos = dict(infos or {})
        extra = dict(infos.get("gd_lab", {}))
        extra["learning_rate"] = self.alg.learning_rate
        if hasattr(self, 'observation_context'):
            extra['observation_context'] = self.observation_context
        if hasattr(self, 'gap_finetune_manifest'):
            extra['gap_finetune'] = self.gap_finetune_manifest
        policy = self.alg.policy
        if isinstance(policy, DreamwaqActorCritic):
            extra["cenet_optimizer_state_dict"] = policy.cenet.optimizer.state_dict()
        if self.deploy_context is not None:
            extra["deploy_context"] = self.deploy_context
        family_columns = getattr(self, "_top5_family_columns", None)
        if family_columns:
            means = capture_family_level_means(self.env.unwrapped.scene.terrain, family_columns)
            if means:
                extra["terrain_level_means_by_family"] = means
                extra["terrain_level_resume_rule"] = "floor_family_mean"
        infos["gd_lab"] = extra
        temporary = str(path) + '.partial'
        super().save(temporary, infos)
        os.replace(temporary, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        extra = (infos or {}).get("gd_lab", {})
        if "learning_rate" in extra:
            self.alg.learning_rate = extra["learning_rate"]
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate
        family_columns = getattr(self, "_top5_family_columns", None)
        if family_columns:
            means = extra.get("terrain_level_means_by_family") or {}
            if not means:
                means = read_family_level_means_from_tensorboard(
                    path, self.current_learning_iteration, family_columns
                )
            if means:
                scene = self.env.unwrapped.scene
                restored = restore_family_start_levels(
                    scene.terrain, scene.env_origins, family_columns, means
                )
                if restored:
                    self.env.reset()
                    print(
                        "[INFO] Resumed terrain levels using floor(family mean): "
                        f"{restored}",
                        flush=True,
                    )
            else:
                print(
                    "[WARN] No saved terrain curriculum levels found; "
                    "using fresh environment initialization",
                    flush=True,
                )
        policy = self.alg.policy
        if load_optimizer and isinstance(policy, DreamwaqActorCritic) and "cenet_optimizer_state_dict" in extra:
            policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
        # The checkpoint was written after finishing ``iter``; continue from the next one.
        self.current_learning_iteration += 1
        return infos

    def export_inference_state(self) -> dict[str, torch.Tensor]:
        self.eval_mode()
        return self.alg.policy.state_dict()
