"""Manifest application includes buffers and auxiliary optimizer gates, not just gradients."""
import hashlib
import yaml
from .migration import NEW_PREFIXES, NEW_KEYS

def checksum(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def frozen_checksums(policy):
    frozen = {n for n, p in policy.named_parameters() if not p.requires_grad}
    for n, _ in policy.named_buffers():
        if not policy.nav_update_old_stats and n.startswith(('actor_obs_normalizer.', 'critic_obs_normalizer.')):
            frozen.add(n)
        if not policy.nav_cenet_step and n.startswith('cenet.'):
            frozen.add(n)
    return {n: checksum(t) for n, t in policy.state_dict().items() if n in frozen}

def assert_frozen(policy, before):
    state = policy.state_dict()
    changed = [n for n, digest in before.items() if checksum(state[n]) != digest]
    if changed:
        raise AssertionError(f'frozen state changed: {changed}')

def apply_manifest(policy, path, *, learning_rate):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    policy.nav_update_old_stats = cfg['old_stats']
    policy.nav_update_new_stats = cfg['new_stats']
    policy.nav_cenet_step = cfg['cenet_step']
    policy.nav_aux_step = cfg['aux_step']
    groups = {'old': [], 'new': []}
    names = {'old': [], 'new': []}
    for name, p in policy.named_parameters():
        p.requires_grad_(any(name.startswith(prefix) for prefix in cfg['train_prefixes']))
        if not p.requires_grad or name.startswith('cenet.'):
            continue
        group = 'new' if name in NEW_KEYS or name.startswith(NEW_PREFIXES) else 'old'
        groups[group].append(p); names[group].append(name)
    result = [{'params': values, 'lr': learning_rate * (cfg['old_lr_scale'] if key == 'old' else 1),
               'gd_lab_group': key} for key, values in groups.items() if values]
    # CENet is decoupled and must never appear in PPO groups.
    if policy.nav_cenet_step:
        for group in policy.cenet.optimizer.param_groups:
            group['lr'] = learning_rate * cfg['old_lr_scale']
    return result, {'stage': cfg['stage'], 'parameter_names': names, 'frozen': frozen_checksums(policy)}
