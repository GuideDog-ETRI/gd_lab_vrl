"""Strict, SHA-bound native Ray+IR -> Nav migration. No source optimizer restoration."""
import hashlib
from pathlib import Path
import torch

NEW_PREFIXES = ('nav_wide_encoder.', 'nav_near_encoder.', 'nav_normalizer.',
                'nav_critic_wide.', 'nav_critic_near.', 'nav_critic_normalizer.')
NEW_KEYS = {'actor.0.W_new', 'critic.0.W_new'}

def migrate_state_dict(source, target):
    extra = set(source) - set(target)
    if extra:
        raise ValueError(f'unexpected source keys: {sorted(extra)}')
    merged, fresh = {}, []
    for key, value in target.items():
        if key in source:
            src = source[key]
            if src.shape != value.shape or src.dtype != value.dtype:
                raise ValueError(f'incompatible tensor {key}: {src.shape}/{src.dtype} -> {value.shape}/{value.dtype}')
            merged[key] = src.detach().clone()
        elif key in NEW_KEYS or key.startswith(NEW_PREFIXES):
            merged[key] = torch.zeros_like(value) if key in NEW_KEYS else value.detach().clone()
            fresh.append(key)
        else:
            raise ValueError(f'unapproved expansion/missing source: {key}')
    return merged, fresh

def migrate_checkpoint(policy, path, *, expected_sha256):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError('3906 source SHA256 mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('iter') != 3906:
        raise ValueError('expected source iteration 3906')
    merged, fresh = migrate_state_dict(checkpoint['model_state_dict'], policy.state_dict())
    policy.load_state_dict(merged, strict=True)
    return {'source': str(path), 'source_sha256': digest, 'source_iteration': 3906,
            'nav_iteration': 0, 'fresh_keys': fresh, 'strict_load': True,
            'ppo_optimizer': 'new', 'cenet_optimizer': 'new'}
