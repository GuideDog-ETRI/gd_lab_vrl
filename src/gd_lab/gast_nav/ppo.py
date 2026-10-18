"""State-selected controller masks for native GAST PPO. CPU/gloo-compatible reductions.
The runner must call act_selected BEFORE executing the selected controller, then
pass nav_controller/nav_executed_action/nav_override_reason to process_env_step.
No proposal-dependent shield is supported.
"""
from pathlib import Path
import torch
import torch.distributed as dist
from gd_lab.gast.teacher import GastPPO, mirror_gast
from gd_lab.gast.geometry import reconstruction_loss
from gd_lab.rl.distributed_sync import average_gradients
from . import schema
from .trainable import frozen_checksums, assert_frozen, apply_manifest

NAV_POLICY = schema.CONTROLLERS.index('NAV_POLICY')
HOLD = schema.CONTROLLERS.index('HOLD_3906')

def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def summed(tensor):
    result = tensor.detach().clone()
    if world_size() > 1: dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result

def masked_global_mean(values, mask):
    values, mask = values.reshape(-1), mask.reshape(-1).bool()
    if values.shape != mask.shape: raise ValueError('mask shape')
    count = summed(mask.sum().to(dtype=torch.float64))
    local = values[mask].sum()
    # local is differentiable even when no elements were selected.
    return local * world_size() / count.clamp_min(1).to(values.dtype), int(count.item())

def normalize_valid(advantages, mask, epsilon):
    a, m = advantages.reshape(-1), mask.reshape(-1).bool()
    count = summed(m.sum().double())
    total = summed(a[m].double().sum())
    mean = total / count.clamp_min(1)
    variance = summed(((a[m].double() - mean) ** 2).sum()) / count.clamp_min(1)
    result = torch.zeros_like(a)
    result[m] = ((a[m].double() - mean) / (variance.sqrt() + epsilon)).to(a.dtype)
    return result.reshape_as(advantages)

def intervention_rewards(rewards, controller, previous, coefficient):
    return rewards + ((controller == HOLD) & (previous != HOLD)).to(rewards.dtype) * coefficient

def mirror_legacy(obs, actions, mode, *, env=None, base_mirror=mirror_gast, privileged_mirror=None):
    """Return only augmented LEGACY rows; caller keeps original batch separately.
    Critic extra geometry must provide its semantic reflection; never silently copy it.
    """
    indices = (mode == schema.MODES.index('LEGACY')).nonzero().flatten()
    selected = obs[indices]
    if not len(indices): return selected, actions[indices], indices
    extra = [k for k in selected.keys() if k.startswith('nav_')]
    augmented, aa = base_mirror(selected.exclude(*extra), actions[indices], env)
    count = len(indices)
    mirrored = augmented[count:]
    for key in extra:
        value = selected[key]
        if key in ('nav_wide', 'nav_near'): value = value.flip(-2)
        elif key == 'nav_features':
            value = value.clone(); value[:, 5] *= -1; value[:, 13] *= -1
        elif key == 'nav_privileged':
            if privileged_mirror is None: raise ValueError('critic privileged mirror contract required')
            value = privileged_mirror(value)
        elif key not in ('nav_controller', 'nav_mode'):
            raise ValueError('unknown mirror channel: ' + key)
        mirrored[key] = value
    return mirrored, aa[count:], indices

class GastNavPPO(GastPPO):
    def __init__(self, *args, nav_config, trainable_manifest=None, **kwargs):
        for key in ('clip_param', 'entropy_coef', 'value_loss_coef', 'gamma', 'lam', 'max_grad_norm'):
            kwargs.setdefault(key, nav_config[key])
        super().__init__(*args, **kwargs)
        if self.schedule != 'fixed': raise ValueError('Nav parameter groups require fixed schedule')
        if self.symmetry is not None: raise ValueError('use mirror_legacy with explicit extra-critic reflection')
        if self.rnd is not None: raise ValueError('RND is outside v1 contract')
        self.configure_trainable(trainable_manifest or Path(__file__).resolve().parents[3] / "configs/gast_nav/trainable_A0.yaml")
        self.nav_config = nav_config
        self.nav_masks = None
        self.nav_controller = None
        self.nav_previous = None
        self.nav_normalization_obs = []

    def configure_trainable(self, manifest):
        groups, report = apply_manifest(self.policy, manifest, learning_rate=self.learning_rate)
        self.optimizer = torch.optim.Adam(groups)
        self.nav_frozen = report["frozen"]
        return report

    def act_selected(self, obs, controller):
        if bool(((controller < 0) | (controller >= len(schema.CONTROLLERS))).any()): raise ValueError('controller')
        self.nav_controller = controller.detach().clone().reshape(-1)
        return super().act(obs)

    def act(self, obs):
        if 'nav_controller' not in obs: raise ValueError('pre-policy controller selection required')
        return self.act_selected(obs, obs['nav_controller'].reshape(-1))

    def process_env_step(self, obs, rewards, dones, extras):
        if self.nav_controller is None: raise RuntimeError('act_selected must precede transition')
        executed = extras['nav_controller'].reshape(-1)
        if not torch.equal(executed, self.nav_controller): raise ValueError('post-proposal controller override forbidden')
        if self.nav_masks is None:
            shape = self.storage.rewards.shape[:2]
            self.nav_masks = torch.zeros(shape, dtype=torch.bool, device=rewards.device)
            self.nav_executed_actions = torch.zeros_like(self.storage.actions)
            self.nav_controllers = torch.zeros(shape, dtype=torch.long, device=rewards.device)
            self.nav_reasons = torch.zeros(shape, dtype=torch.long, device=rewards.device)
            self.nav_previous = torch.full_like(executed, NAV_POLICY)
        nav_rows = executed == NAV_POLICY
        if not torch.equal(extras['nav_executed_action'][nav_rows], self.transition.actions[nav_rows]):
            raise ValueError('NAV_POLICY proposal must be the executed action')
        step = self.storage.step
        self.nav_masks[step] = executed == NAV_POLICY
        self.nav_controllers[step] = executed
        self.nav_executed_actions[step] = extras['nav_executed_action']
        self.nav_reasons[step] = extras['nav_override_reason']
        adjusted = intervention_rewards(rewards, executed, self.nav_previous, self.nav_config['intervention_reward'])
        self.nav_previous = torch.where(dones.reshape(-1).bool(), NAV_POLICY, executed).detach().clone()
        # Keep normalization immutable within rollout+PPO replay; update moments only afterwards.
        self.nav_normalization_obs.append(obs.detach().clone())
        flags = self.policy.nav_update_old_stats, self.policy.nav_update_new_stats
        self.policy.nav_update_old_stats = self.policy.nav_update_new_stats = False
        try: super().process_env_step(obs, adjusted, dones, extras)
        finally: self.policy.nav_update_old_stats, self.policy.nav_update_new_stats = flags
        self.nav_controller = None

    def compute_returns(self, obs):
        self.storage.compute_returns(self.policy.evaluate(obs).detach(), self.gamma, self.lam, normalize_advantage=False)
        self.storage.advantages.copy_(normalize_valid(self.storage.advantages, self.nav_masks,
                                                     self.nav_config['advantage_epsilon']))

    def update(self):
        st, p = self.storage, self.policy
        if self.nav_masks is None: raise RuntimeError('no selected-controller rollout')
        before = self.nav_frozen
        assert_frozen(p, before)
        obs = st.observations.flatten(0, 1)
        flat = lambda x: x.flatten(0, 1)
        actions, old_lp = flat(st.actions), flat(st.actions_log_prob).reshape(-1)
        old_v, returns = flat(st.values).reshape(-1), flat(st.returns).reshape(-1)
        advantage, mask = flat(st.advantages).reshape(-1), self.nav_masks.flatten()
        old_mu, old_sigma = flat(st.mu), flat(st.sigma)
        metrics = {'actor_skipped': 0, 'surrogate': 0., 'value': 0., 'kl': 0.}
        updates = 0
        for _ in range(self.num_learning_epochs):
            for ids in torch.tensor_split(torch.randperm(len(actions), device=actions.device), self.num_mini_batches):
                batch = obs[ids]
                p.act(batch)
                lp = p.get_actions_log_prob(actions[ids])
                ratio = torch.exp(lp - old_lp[ids])
                raw = torch.maximum(-advantage[ids] * ratio,
                                    -advantage[ids] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param))
                actor, count = masked_global_mean(raw, mask[ids])
                entropy, _ = masked_global_mean(p.entropy, mask[ids])
                with torch.no_grad():
                    kl_raw = (torch.log(p.action_std / old_sigma[ids]) +
                              (old_sigma[ids].square() + (old_mu[ids] - p.action_mean).square()) /
                              (2 * p.action_std.square()) - .5).sum(-1)
                    kl, _ = masked_global_mean(kl_raw, mask[ids])
                value = p.evaluate(batch).reshape(-1)
                vf = (value - returns[ids]).square()
                if self.use_clipped_value_loss:
                    clipped = old_v[ids] + (value - old_v[ids]).clamp(-self.clip_param, self.clip_param)
                    vf = torch.maximum(vf, (clipped - returns[ids]).square())
                critic, _ = masked_global_mean(vf, torch.ones_like(vf, dtype=torch.bool))
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.value_loss_coef * critic
                if count: loss = loss + actor - self.entropy_coef * entropy
                else: metrics['actor_skipped'] += 1
                loss.backward()
                average_gradients(p.parameters())
                torch.nn.utils.clip_grad_norm_([v for v in p.parameters() if v.requires_grad], self.max_grad_norm)
                self._finite_step()
                updates += 1
                for name, val in [('surrogate', actor), ('value', critic), ('kl', kl)]: metrics[name] += (summed(val) / world_size()).item()
        # Original auxiliary semantics: all controllers, terrain reconstruction not actor-masked.
        if p.nav_cenet_step:
            p.cenet.update_bootstrap_from_episode_rewards(self._global_episode_returns())
            metrics.update(self._update_cenet())
            p.notify_iteration_done()
        if p.nav_aux_step:
            ids = torch.randperm(len(actions), device=actions.device)[:self.nav_config['auxiliary_samples']]
            pred = p.terrain_decoder(p.terrain_encoder(obs['gast_history'][ids])).reshape(-1, 187, 6)
            aux = reconstruction_loss(pred, obs['gast_clean'][ids])
            self.optimizer.zero_grad(set_to_none=True)
            (self.nav_config['auxiliary_coef'] * aux).backward()
            average_gradients(p.parameters())
            torch.nn.utils.clip_grad_norm_(p.parameters(), self.max_grad_norm)
            self._finite_step()
            metrics['gast_reconstruction'] = aux.item()
        for snapshot in self.nav_normalization_obs: p.update_normalization(snapshot)
        self.nav_normalization_obs.clear()
        assert_frozen(p, before)
        st.clear()
        for name in ('surrogate', 'value', 'kl'): metrics[name] /= max(updates, 1)
        return metrics

    def _finite_step(self):
        from gd_lab.rl.distributed_sync import any_rank
        bad = any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in self.policy.parameters())
        if any_rank(bad, self.device):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('nonfinite Nav gradient; no optimizer step on any rank')
        self.optimizer.step()
