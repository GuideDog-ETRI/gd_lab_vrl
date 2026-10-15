"""PPO with a decoupled CENet auxiliary update, on top of upstream rsl_rl PPO."""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .actor_critic import DreamwaqActorCritic


class DreamwaqPPO(PPO):
    """Standard PPO plus, after each PPO update, a CENet pass over the same rollout.

    The CENet owns its optimizer, so PPO and CENet updates are independent; running
    the CENet pass after (rather than interleaved with) the PPO minibatch loop keeps
    the update-forward code identical to the rollout policy within one iteration.
    """

    policy: DreamwaqActorCritic

    def __init__(
        self,
        *args: Any,
        episode_reward_window: int = 100,
        min_learning_rate: float = 1.0e-5,
        **kwargs: Any,
    ) -> None:
        self._min_learning_rate = float(min_learning_rate)
        super().__init__(*args, **kwargs)
        if not isinstance(self.policy, DreamwaqActorCritic):
            raise TypeError(f"DreamwaqPPO requires a DreamwaqActorCritic policy, got {type(self.policy).__name__}")
        # Adam was built from the raw argument, before the floor applied.
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self._episode_returns: deque[float] = deque(maxlen=episode_reward_window)
        self._cur_return: torch.Tensor | None = None
        self._next_latest: torch.Tensor | None = None
        self._aux_dones: torch.Tensor | None = None

    # -- adaptive learning-rate floor --------------------------------------
    # Upstream hardcodes ``max(1e-5, lr / 1.5)`` inside the minibatch loop, so
    # raising the floor means either copying that method or intercepting the
    # assignment. The stock line writes through this setter and the
    # ``param_group["lr"]`` line after it reads the clamped value back; clamping
    # after ``update()`` would leave that update's minibatches below the floor.
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        # ``PPO.__init__`` writes this before the subclass finishes.
        self._learning_rate = max(getattr(self, "_min_learning_rate", 0.0), float(value))

    # -- rollout bookkeeping ----------------------------------------------
    def _ensure_aux_buffers(self, rewards: torch.Tensor) -> None:
        if self._next_latest is None:
            st = self.storage
            self._next_latest = torch.zeros(
                st.num_transitions_per_env, st.num_envs, self.policy.one_step_obs_dim, device=self.device
            )
            self._aux_dones = torch.zeros(st.num_transitions_per_env, st.num_envs, dtype=torch.bool, device=self.device)
            self._cur_return = torch.zeros(st.num_envs, device=self.device)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self._ensure_aux_buffers(rewards)
        step = self.storage.step
        policy_group = self.policy.obs_groups["policy"][0]
        # ``obs`` is o_{t+1}: its latest one-step slice is the CENet reconstruction target.
        self._next_latest[step].copy_(obs[policy_group][..., self.policy.latest_idx])
        done_rows = dones.view(-1).bool()
        self._aux_dones[step].copy_(done_rows)

        self._cur_return += rewards.view(-1)
        if done_rows.any():
            self._episode_returns.extend(self._cur_return[done_rows].tolist())
            self._cur_return[done_rows] = 0.0

        super().process_env_step(obs, rewards, dones, extras)

    # -- non-finite step guard -------------------------------------------
    # 2026-10-08 22:18 the 3-GPU run died at iteration 8016: a non-finite PPO gradient, averaged into every
    # rank by reduce_parameters, made the scalar action std NaN and the next minibatch's sample() raised.
    # Each PPO optimizer step is now taken only if every rank agrees the (already averaged and clipped)
    # gradients are finite; otherwise the step is skipped on all ranks (no Adam moment/step-count change).
    # After a step, non-finite parameters roll the parameters AND optimizer state back to before the step.
    # The scalar std is kept >= ``min_action_std``. Too many consecutive skips stop the run on all ranks.
    min_action_std = 0.01
    max_consecutive_guard_skips = 50

    def _install_step_guard(self) -> None:
        optimizer = self.optimizer
        if getattr(optimizer, "_gd_step_guard", False):
            return
        from .distributed import any_rank
        original = optimizer.step

        def guarded_step(*args: Any, **kwargs: Any):
            params = [p for group in optimizer.param_groups for p in group["params"]]
            bad_grad = any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params)
            if any_rank(bad_grad, self.device):
                optimizer.zero_grad(set_to_none=True)
                self._guard_skipped += 1
                self._guard_consecutive += 1
                if self._guard_consecutive > self.max_consecutive_guard_skips:
                    raise RuntimeError(f"{self._guard_consecutive} consecutive non-finite PPO steps; stopping")
                print(f"[GUARD] skipped PPO optimizer step (non-finite gradient), total {self._guard_skipped}",
                      flush=True)
                return None
            saved_params = [p.detach().clone() for p in params]
            saved_state = copy.deepcopy(optimizer.state_dict())
            out = original(*args, **kwargs)
            bad_param = any(not bool(torch.isfinite(p).all()) for p in params)
            # Adam moments can go non-finite before the parameters do.
            bad_state = any(torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())
                            for state in optimizer.state.values() for v in state.values())
            if any_rank(bad_param or bad_state, self.device):
                with torch.no_grad():
                    for p, saved in zip(params, saved_params):
                        p.copy_(saved)
                optimizer.load_state_dict(saved_state)
                self._guard_rolled_back += 1
                self._guard_consecutive += 1
                if self._guard_consecutive > self.max_consecutive_guard_skips:
                    raise RuntimeError(f"{self._guard_consecutive} consecutive non-finite PPO steps; stopping")
                print(f"[GUARD] rolled back optimizer step (non-finite parameters/state), total "
                      f"{self._guard_rolled_back}", flush=True)
                return None
            # Clamp only after the finite check, so a -inf std is caught rather than hidden.
            std = getattr(self.policy, "std", None)
            if isinstance(std, torch.nn.Parameter) and getattr(self.policy, "noise_std_type", "scalar") == "scalar":
                with torch.no_grad():
                    std.clamp_(min=self.min_action_std)
            self._guard_consecutive = 0
            return out

        optimizer.step = guarded_step
        optimizer._gd_step_guard = True

    # -- update ------------------------------------------------------------
    def update(self) -> dict[str, float]:
        from .distributed import active
        import torch.distributed as dist
        for name in ("_guard_skipped", "_guard_rolled_back", "_guard_consecutive"):
            if not hasattr(self, name):
                setattr(self, name, 0)
        skipped0, rolled0 = self._guard_skipped, self._guard_rolled_back
        # The optimizer can be replaced after construction (resume rebuilds the grouped optimizer).
        self._install_step_guard()
        returns = list(self._episode_returns)
        if active():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, returns)
            returns = [value for batch in gathered for value in batch]
        self.policy.cenet.update_bootstrap_from_episode_rewards(returns)
        loss_dict = super().update()
        loss_dict["guard_skipped_steps"] = float(self._guard_skipped - skipped0)
        loss_dict["guard_rolled_back_steps"] = float(self._guard_rolled_back - rolled0)
        loss_dict.update(self._update_cenet())
        self.policy.notify_iteration_done()
        if active():
            for buffer in self.policy.cenet.buffers():
                dist.broadcast(buffer, src=0)
        return loss_dict

    def _update_cenet(self) -> dict[str, float]:
        st = self.storage
        policy_group = self.policy.obs_groups["policy"][0]
        critic_group = self.policy.obs_groups["critic"][0]
        # storage.clear() only resets the write cursor; the tensors still hold
        # this iteration's rollout.
        history_raw = st.observations[policy_group].flatten(0, 1)
        critic_raw = st.observations[critic_group].flatten(0, 1)
        next_latest = self._next_latest.flatten(0, 1)
        valid = ~self._aux_dones.flatten(0, 1)

        batch_size = history_raw.shape[0]
        num_batches = self.num_mini_batches
        mini_batch_size = batch_size // num_batches
        totals: dict[str, float] = {}
        num_updates = 0
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(num_batches * mini_batch_size, device=self.device)
            for i in range(num_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                history = self.policy.actor_obs_normalizer(history_raw[idx])
                velocity_target = critic_raw[idx][..., self.policy._velocity_target_slice]
                recon_target = self.policy.normalize_latest(next_latest[idx])
                metrics = self.policy.cenet.update(
                    obs_history=history,
                    velocity_target=velocity_target,
                    next_obs_target=recon_target,
                    valid_mask=valid[idx],
                )
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                num_updates += 1
        return {key: value / num_updates for key, value in totals.items()}
