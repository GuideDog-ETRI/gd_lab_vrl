"""Unchanged CENet + terrain denoising autoencoder and PPO."""
import torch
from torch import nn
from isaaclab.utils import configclass
from gd_lab.teachers.cvtt.actor_critic import DreamwaqVrlActorCritic
from gd_lab.teachers.cvtt.agent_cfg import DreamwaqVrlRunnerCfg
from gd_lab.rl.ppo import DreamwaqPPO
from gd_lab.gast.geometry import reconstruction_loss
from gd_lab.gast.temporal import TemporalTerrainEncoder
from gd_lab.rl.distributed import average_gradients, update_normalizer
from gd_lab.methods.dreamwaq.vrl_symmetry import mirror_vrl_observations


class GastActorCritic(DreamwaqVrlActorCritic):
    _nonfinite_obs_reports = 0

    def update_normalization(self, obs):
        # One NaN row in a raw all-reduced sum poisons every rank's normalizer and,
        # through it, all actions. Keep such rows out and report where they came from.
        if self.actor_obs_normalization:
            update_normalizer(self.actor_obs_normalizer, self._finite_rows('actor', self.get_actor_obs(obs)))
        if self.critic_obs_normalization:
            update_normalizer(self.critic_obs_normalizer, self._finite_rows('critic', self.get_critic_obs(obs)))

    def _finite_rows(self, group, values):
        finite = torch.isfinite(values).all(-1)
        if bool(finite.all()):
            return values
        if self._nonfinite_obs_reports < 20:
            self._nonfinite_obs_reports += 1
            from gd_lab.rl.distributed import active
            import torch.distributed as dist
            bad = (~finite).nonzero().flatten()
            columns = (~torch.isfinite(values[bad])).any(0).nonzero().flatten()
            print(f'[GAST_NONFINITE_OBS] rank={dist.get_rank() if active() else 0} group={group} '
                  f'envs={bad[:16].tolist()} n_envs={bad.numel()} columns={columns[:32].tolist()}', flush=True)
        return values[finite]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terrain_encoder = TemporalTerrainEncoder()
        self.terrain_decoder = nn.Sequential(nn.Linear(32, 128), nn.ELU(), nn.Linear(128, 187*6))

    def terrain_latent(self, obs):
        terrain = self.normalized_height_scan(obs)
        latent = self.terrain_encoder(obs['gast_history'])
        # Exact zero terrain contribution during no-scan segments.
        available = (terrain[:, 187:].sum(-1, keepdim=True) > 0).to(latent.dtype)
        # where(), not multiply: NaN*0 is NaN.
        return torch.where(available > 0, latent, torch.zeros_like(latent))


def mirror_gast(obs=None, actions=None, env=None):
    base = obs.exclude('gast_clean', 'gast_history') if obs is not None else None
    augmented, action_aug = mirror_vrl_observations(base, actions, env)
    if obs is not None:
        clean = obs['gast_clean']
        augmented['gast_clean'] = torch.cat((clean, clean.reshape(-1, 4, 11, 17).flip(-2).reshape_as(clean)), 0)
        history = obs['gast_history'].reshape(-1, 8, 375)
        mirrored = history.clone()
        mirrored[..., :374] = history[..., :374].reshape(-1, 8, 2, 11, 17).flip(-2).reshape(-1, 8, 374)
        augmented['gast_history'] = torch.cat((history.flatten(1), mirrored.flatten(1)), 0)
    return augmented, action_aug


def _nonfinite_rows(tensor):
    """Bool [steps, envs] of rows holding any NaN/inf, without an element-sized temporary."""
    if not tensor.is_floating_point():
        return None
    # NaN/inf in any element makes the row sum non-finite; inputs are bounded, so
    # a finite row cannot overflow to inf.
    return ~torch.isfinite(tensor.flatten(2).sum(-1, dtype=torch.float64))


def _state_finite(state):
    if torch.is_tensor(state):
        return not state.is_floating_point() or bool(torch.isfinite(state).all())
    if isinstance(state, dict):
        return all(_state_finite(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return all(_state_finite(v) for v in state)
    return True


def _clone_state(state):
    if torch.is_tensor(state):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: _clone_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_clone_state(v) for v in state)
    return state


class GastPPO(DreamwaqPPO):
    """PPO + terrain auxiliary update, rolled back as a whole if it goes non-finite.

    Arm4 died twice (iter 8727, 10342) on the auxiliary clip's non-finite check
    with every logged loss, Adam moment and checkpoint finite just before. That
    clip only detects; rsl_rl's PPO clip silently propagates NaN. One bad update
    is therefore undone (policy, normalizers, CENet, both optimizers restored to
    the last good update) and the source is logged; repeated ones still stop.
    """
    max_nonfinite_skips = 3
    # Non-consecutive skips are capped too: a periodic NaN source must not be
    # silently skipped forever.
    max_nonfinite_skips_per_1000 = 10

    def process_env_step(self, obs, rewards, dones, extras):
        # An env whose physics went non-finite is terminated (tasks.nonfinite_state)
        # and reset before its next observation; only its last reward is unusable.
        finite = torch.isfinite(rewards)
        if not bool(finite.all()):
            print(f'[GAST_NONFINITE_REWARD] envs={(~finite).flatten().nonzero().flatten()[:16].tolist()}', flush=True)
            rewards = torch.where(finite, rewards, torch.zeros_like(rewards))
        super().process_env_step(obs, rewards, dones, extras)

    def _snapshot(self):
        return (_clone_state(self.policy.state_dict()), _clone_state(self.optimizer.state_dict()),
                _clone_state(self.policy.cenet.optimizer.state_dict()), self.learning_rate)

    def _restore(self, snapshot):
        policy, optimizer, cenet_optimizer, learning_rate = snapshot
        self.policy.load_state_dict(policy)
        # load_state_dict reuses same-device tensors: load clones so later in-place
        # Adam steps cannot write NaN back into the snapshot.
        self.optimizer.load_state_dict(_clone_state(optimizer))
        self.policy.cenet.optimizer.load_state_dict(_clone_state(cenet_optimizer))
        self.learning_rate = learning_rate
        from gd_lab.gast.warm_start import is_grouped
        if not is_grouped(self.optimizer):  # per-group rates were restored by load_state_dict above
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        self.policy.cenet.optimizer.zero_grad(set_to_none=True)

    def _storage_report(self):
        st = self.storage
        tensors = {f'obs/{k}': v for k, v in st.observations.items()}
        for name in ('actions', 'rewards', 'values', 'returns', 'advantages', 'actions_log_prob', 'mu', 'sigma'):
            value = getattr(st, name, None)
            if torch.is_tensor(value):
                tensors[name] = value
        bad, envs = {}, None
        for key, value in tensors.items():
            rows = _nonfinite_rows(value)
            if rows is not None and bool(rows.any()):
                bad[key] = int(rows.sum())  # (step, env) rows
                first_step = rows.any(1).nonzero()[0].item()
                bad[key + '/first_step'] = first_step
                envs = rows.any(0) if envs is None else envs | rows.any(0)
        if envs is not None:
            bad['env_ids'] = envs.nonzero().flatten()[:16].tolist()
        return bad

    def _fail(self, stage, storage_report):
        from gd_lab.rl.distributed import active
        import torch.distributed as dist
        reports = [storage_report]
        if active():
            reports = [None] * dist.get_world_size()
            dist.all_gather_object(reports, storage_report)
        self._consecutive_nonfinite = getattr(self, '_consecutive_nonfinite', 0) + 1
        self._skip_updates = getattr(self, '_skip_updates', []) + [self._update_count]
        recent = [u for u in self._skip_updates if u > self._update_count - 1000]
        self._skip_updates = recent
        self._restore(self._last_good)
        message = (f'[GAST_NONFINITE] stage={stage} consecutive={self._consecutive_nonfinite} '
                   f'storage_by_rank={reports}')
        if not active() or dist.get_rank() == 0:
            print(message, flush=True)
        if (self._consecutive_nonfinite > self.max_nonfinite_skips
                or len(recent) > self.max_nonfinite_skips_per_1000):
            raise RuntimeError(f'{message}; rolled back to last good update, giving up')

    def update(self):
        from gd_lab.rl.distributed import any_rank
        self._update_count = getattr(self, '_update_count', 0) + 1
        storage_report = self._storage_report()
        # Normalizers were updated during this rollout; NaN data would poison them.
        rollout_bad = any_rank(storage_report or not _state_finite(self.policy.state_dict()), self.device)
        if not hasattr(self, '_last_good'):
            if rollout_bad:
                raise RuntimeError(f'[GAST_NONFINITE] stage=rollout before any good update: {storage_report}')
            self._last_good = self._snapshot()
            if not _state_finite(self._last_good):
                raise RuntimeError('[GAST_NONFINITE] initial snapshot is not finite')
        if rollout_bad:
            self.storage.clear()
            self._fail('rollout', storage_report)
            return {'gast_nonfinite_skip': 1.}
        metrics = super().update()
        if any_rank(not _state_finite(self.policy.state_dict()), self.device):
            self._fail('ppo_or_cenet', storage_report)
            return {**metrics, 'gast_nonfinite_skip': 1.}
        terrain = self.storage.observations['gast_history'].flatten(0, 1)
        clean = self.storage.observations['gast_clean'].flatten(0, 1)
        # A bounded auxiliary pass, using the SAME Adam state saved by the runner.
        idx = torch.randperm(terrain.shape[0], device=terrain.device)[:2048]
        encoded = self.policy.terrain_encoder(terrain[idx])
        pred = self.policy.terrain_decoder(encoded).reshape(-1, 187, 6)
        loss = reconstruction_loss(pred, clean[idx])
        self.optimizer.zero_grad()
        (.1 * loss).backward()
        average_gradients(self.policy.parameters())
        norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
        if any_rank(not (torch.isfinite(loss) and torch.isfinite(norm)), self.device):
            self._fail(f'terrain_aux(loss={loss.item():.4g},grad_norm={norm.item():.4g},'
                       f'pred_nonfinite={int((~torch.isfinite(pred)).sum())})', storage_report)
            return {**metrics, 'gast_nonfinite_skip': 1.}
        self.optimizer.step()
        self._consecutive_nonfinite = 0
        self._last_good = self._snapshot()
        metrics['gast_reconstruction'] = loss.item()
        metrics['gast_nonfinite_skip'] = 0.
        return metrics


@configclass
class GastRunnerCfg(DreamwaqVrlRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = 'gd_lab.gast.teacher:GastActorCritic'
        self.algorithm.class_name = 'gd_lab.gast.teacher:GastPPO'
        self.algorithm.symmetry_cfg.data_augmentation_func = 'gd_lab.gast.teacher:mirror_gast'
        self.logger = 'tensorboard'
        self.experiment_name = 'gast/arm4'
