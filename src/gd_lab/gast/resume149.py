"""Strict UPDATE149B source validation, independent of Isaac/GPU."""
import hashlib
import torch


def validate_resume(path, sha, task):
    with open(path, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != sha:
        raise ValueError('UPDATE149B source SHA mismatch')
    c = torch.load(path, map_location='cpu', weights_only=False)
    e = c['infos']['gd_lab']
    if (e.get('gast_warm_start') or {}).get('task') != task:
        raise ValueError('UPDATE149B source task mismatch')
    for o in (c.get('optimizer_state_dict'), e.get('cenet_optimizer_state_dict')):
        if not o or not o.get('state') or not o.get('param_groups'):
            raise ValueError('UPDATE149B missing optimizer state; no silent warm start')
    if not e.get('terrain_level_means_by_family') or not isinstance(c.get('iter'), int):
        raise ValueError('UPDATE149B missing iteration/terrain levels')
    steps = e['total_timesteps'] // e['total_envs']
    if steps != (c['iter'] + 1) * 100:
        raise ValueError('UPDATE149B unexpected source horizon/progress')
    groups = c['optimizer_state_dict']['param_groups']
    if {g.get('gd_lab_group'):g['lr'] for g in groups} != {'policy':1e-4,'terrain':1e-3}:
        raise ValueError('UPDATE149B unexpected PPO groups/LR')
    def summary(o):
        steps = [float(v['step']) for v in o['state'].values() if 'step' in v]
        return dict(state_count=len(o['state']), step_min=min(steps), step_max=max(steps),
                    lrs=[g['lr'] for g in o['param_groups']])
    return dict(source=str(path), source_sha256=actual, task=task, source_iteration=c['iter'],
        start_iteration=c['iter']+1, absolute_step_offset=steps,
        ppo=summary(c['optimizer_state_dict']), cenet=summary(e['cenet_optimizer_state_dict']),
        terrain_level_means_by_family=e['terrain_level_means_by_family'])


def verify_loaded(alg, path):
    """Fail on any model or optimizer tensor/parameter-group mismatch after runner.load."""
    c = torch.load(path, map_location='cpu', weights_only=False)
    def equal(actual, expected, label):
        if torch.is_tensor(expected):
            if not torch.equal(actual.detach().cpu(), expected):
                raise ValueError(f'UPDATE149B state differs: {label}')
        elif isinstance(expected, dict):
            if actual.keys() != expected.keys():
                raise ValueError(f'UPDATE149B keys differ: {label}')
            for k in expected: equal(actual[k], expected[k], f'{label}.{k}')
        elif isinstance(expected, (list, tuple)):
            if len(actual) != len(expected): raise ValueError(f'UPDATE149B length differs: {label}')
            for i,(a,b) in enumerate(zip(actual, expected)): equal(a,b,f'{label}.{i}')
        elif actual != expected:
            raise ValueError(f'UPDATE149B value differs: {label}')
    equal(alg.policy.state_dict(), c['model_state_dict'], 'model')
    equal(alg.optimizer.state_dict(), c['optimizer_state_dict'], 'PPO')
    equal(alg.policy.cenet.optimizer.state_dict(), c['infos']['gd_lab']['cenet_optimizer_state_dict'], 'CENet')
    return dict(model_exact=True, ppo_exact=True, cenet_exact=True)


def reset_restored_env(env):
    """Reset pose/history without letting an artificial reset advance restored curriculum."""
    raw = env.unwrapped
    levels = raw.scene.terrain.terrain_levels.clone()
    origins = raw.scene.env_origins.clone()
    compute = raw.curriculum_manager.compute
    try:
        raw.curriculum_manager.compute = lambda *args, **kwargs: None
        env.reset()
    finally:
        raw.curriculum_manager.compute = compute
    if not torch.equal(levels, raw.scene.terrain.terrain_levels) or not torch.equal(origins, raw.scene.env_origins):
        raise ValueError('UPDATE149B reset changed restored levels/origins')
