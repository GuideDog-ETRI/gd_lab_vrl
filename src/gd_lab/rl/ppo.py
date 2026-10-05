"""PPO with a decoupled CENet auxiliary update, on top of upstream rsl_rl PPO."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.distributed as dist
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .actor_critic import DreamwaqActorCritic
from .distributed_sync import (
    assert_synchronized_state,
    average_gradients,
    distributed_active,
    install_collective_optimizer_step,
    raise_if_any,
)


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
        distributed_check_interval: int = 100,
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
        install_collective_optimizer_step(self.optimizer, device=self.device, context="PPO optimizer")
        install_collective_optimizer_step(self.policy.cenet.optimizer, device=self.device, context="CENet optimizer")
        self.distributed_check_interval = max(1, int(distributed_check_interval))
        self._distributed_update_count = 0
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

    # -- update ------------------------------------------------------------
    def reduce_parameters(self) -> None:
        """Average PPO actor/critic/terrain gradients after rank-wide finite checks."""
        parameters = list(self.policy.parameters())
        if self.rnd is not None:
            parameters.extend(self.rnd.parameters())
        nonfinite = any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        raise_if_any(
            "non-finite PPO gradient" if nonfinite else None,
            device=self.device,
            context="PPO gradients",
        )
        average_gradients(parameters)

    def _global_episode_returns(self) -> list[float]:
        local = list(self._episode_returns)
        if not distributed_active():
            return local
        gathered: list[list[float] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        return [reward for rank_rewards in gathered for reward in (rank_rewards or [])]

    def update(self) -> dict[str, float]:
        self.policy.cenet.update_bootstrap_from_episode_rewards(self._global_episode_returns())
        loss_dict = super().update()
        loss_dict.update(self._update_cenet())
        self.policy.notify_iteration_done()
        self._distributed_update_count += 1
        if self._distributed_update_count % self.distributed_check_interval == 0:
            assert_synchronized_state(
                self.policy,
                [("ppo", self.optimizer), ("cenet", self.policy.cenet.optimizer)],
                context=f"PPO/CENet update {self._distributed_update_count}",
            )
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
