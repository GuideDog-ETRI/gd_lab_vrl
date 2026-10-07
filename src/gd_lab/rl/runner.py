"""On-policy runner with importlib class resolution and full resume state."""

from __future__ import annotations

import importlib
import json
import os
import signal
from pathlib import Path

import torch
import torch.distributed as dist
from rsl_rl.algorithms import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .actor_critic import DreamwaqActorCritic
from .distributed_sync import assert_synchronized_state, raise_if_any
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
        count = torch.tensor(self.env.num_envs, device=self.device)
        if self.is_distributed:
            dist.all_reduce(count)
        self._global_env_count = int(count.item())

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
        # Two ways to stop with a save: a stop file (GD_LAB_STOP_FILE, read by rank 0) or SIGUSR1 to any rank.
        signal_stop = False
        self.stopped_early = False
        class SavedStop(Exception):
            pass

        def request_stop(signum, frame):
            nonlocal signal_stop
            signal_stop = True
        previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGUSR1,)}
        for s in previous_handlers:
            signal.signal(s, request_stop)

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
            # GAST teacher terrain diagnostics (blackout, valid cells, noise strength), rank-summed.
            terrain_term = getattr(self.env.unwrapped, '_gast_terrain_term', None)
            if terrain_term is not None and hasattr(terrain_term, 'diagnostic_counts'):
                counts = terrain_term.diagnostic_counts.clone()
                terrain_term.diagnostic_counts.zero_()
                if self.is_distributed:
                    dist.all_reduce(counts)
                result.update(gast_blackout_fraction=(counts[0]/counts[1].clamp_min(1)).item(),
                              gast_valid_cell_fraction=(counts[2]/counts[3].clamp_min(1)).item(),
                              gast_noise_strength=(counts[4]/counts[1].clamp_min(1)).item())
            self._latest_diagnostics = {k: float(v) for k, v in result.items()}
            status = {"selected": False, "saved": False, "error": None}
            if rank == 0 and not self.disable_logs and self.log_dir is not None:
                criteria, criteria_message = self._top5_criteria_reloader.refresh()
                if criteria_message:
                    level = "INFO" if criteria_message.startswith("Reloaded") else "WARN"
                    print(f"[{level}] {criteria_message}", flush=True)
                status = rank_and_save_top5(
                    rank, episodes, Path(self.log_dir) / "best_top5", next_iteration,
                    criteria["top5_min_spacing"], self.save, criteria=criteria,
                    diagnostics=self._latest_diagnostics,
                )
                with open(Path(self.log_dir) / 'top5_decisions.jsonl', 'a') as stream:
                    stream.write(json.dumps({'iteration': next_iteration, **status}, default=str) + '\n')
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
                flag = torch.tensor(int(signal_stop), device=self.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                requested = requested or bool(flag.item())
            else:
                requested = requested or signal_stop
            if requested:
                self.stopped_early = True
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
            for s, handler in previous_handlers.items():
                signal.signal(s, handler)

    def save(self, path: str, infos: dict | None = None) -> None:
        infos = dict(infos or {})
        extra = dict(infos.get("gd_lab", {}))
        extra["learning_rate"] = self.alg.learning_rate
        if hasattr(self, 'observation_context'):
            extra['observation_context'] = self.observation_context
        if hasattr(self, 'gap_finetune_manifest'):
            extra['gap_finetune'] = self.gap_finetune_manifest
        extra["training_diagnostics"] = getattr(self, '_latest_diagnostics', {})
        extra["total_envs"] = getattr(self, '_global_env_count', self.env.num_envs)
        extra["total_timesteps"] = (self.current_learning_iteration + 1) * extra["total_envs"] * self.num_steps_per_env
        if getattr(self, "gast_warm_start", None):
            extra["gast_warm_start"] = self.gast_warm_start
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
        infos = None
        error = None
        try:
            infos = self._load_local(path, load_optimizer=load_optimizer, map_location=map_location)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        raise_if_any(error, device=self.device, context=f"checkpoint resume {path}")
        policy = self.alg.policy
        optimizers = [("ppo", self.alg.optimizer)]
        if hasattr(policy, "cenet") and hasattr(policy.cenet, "optimizer"):
            optimizers.append(("cenet", policy.cenet.optimizer))
        assert_synchronized_state(policy, optimizers, context=f"checkpoint resume {path}")
        return infos

    def _load_local(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        extra = (infos or {}).get("gd_lab", {})
        if "learning_rate" in extra:
            self.alg.learning_rate = extra["learning_rate"]
            # A grouped optimizer (GAST warm start: backbone vs terrain encoder) keeps the per-group
            # rates its saved state restored; only a single-rate optimizer is reset to the saved rate.
            if not any("gd_lab_group" in group for group in self.alg.optimizer.param_groups):
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

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Upstream assumes identical rank sizes. Correct the cumulative count
        # when e.g. 4096 environments are split 1366/1365/1365.
        total_envs = getattr(self, '_global_env_count', self.env.num_envs * self.gpu_world_size)
        actual = self.num_steps_per_env * total_envs
        assumed = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps = (locs['it'] + 1) * actual - assumed
        super().log(locs, width, pad)
        elapsed = locs['collection_time'] + locs['learn_time']
        print(f'[TRAIN_PROGRESS] iteration={locs["it"]} total_envs={total_envs} '
              f'timesteps={self.tot_timesteps} seconds={elapsed:.3f} '
              f'remaining_seconds_estimate={elapsed * (locs["tot_iter"]-locs["it"]-1):.0f}', flush=True)
