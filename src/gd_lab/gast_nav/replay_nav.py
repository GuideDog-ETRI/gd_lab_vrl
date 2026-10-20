"""RL replay recording for GAST-Nav with tools/rl_replay RolloutLog (2026-10-21).

The Isaac RewardManager sees the whole Nav reward as one term, ``nav_total``. This
recorder replaces that column with the Nav terms (progress, stop_success, ...), so
the viewer's per-term bars and returns show which Nav reward drives a behaviour.
Columns still sum to the env reward.

It also writes ``<out>.stop_stats.json``: over every env and step where a stop is
required (NAV/TURN with go=0 or stair-end wait), the mean per-term reward rate when
the robot is stopped vs. still moving. A term that pays more while moving pulls the
policy away from stopping.

Environment variables (all optional except the output path):
  GAST_NAV_REPLAY_OUT          output JSON path (required; must not exist)
  GAST_NAV_REPLAY_STEPS        recorded steps (default 1000 = 10 s)
  GAST_NAV_REPLAY_WARMUP       steps before recording (default 200)
  GAST_NAV_REPLAY_KEEP_ENVS    envs written to the JSON (default 8); stats use all envs
  GAST_NAV_REPLAY_TOOLS        tools/rl_replay directory
"""
import json
import os
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

TASKS = ('', 'passage', 'turn', 'stairs', 'stop_start', 'legacy')


class _RateShim:
    """Stands in for RewardManager inside RolloutLog: names at init, rates at after_step."""

    def __init__(self, names):
        self.active_terms = names
        self._step_reward = None


def record_nav_replay(*, env, policy, obs, args, root):
    tools = Path(os.environ.get('GAST_NAV_REPLAY_TOOLS', '/home/user/gd_project/gd_lab_vrl/tools/rl_replay'))
    sys.path.insert(0, str(tools))
    from rollout_log import RolloutLog

    out = Path(os.environ['GAST_NAV_REPLAY_OUT'])
    steps = int(os.environ.get('GAST_NAV_REPLAY_STEPS', '1000'))
    warmup = int(os.environ.get('GAST_NAV_REPLAY_WARMUP', '200'))
    keep = int(os.environ.get('GAST_NAV_REPLAY_KEEP_ENVS', '8'))
    if args.resume is None:
        raise ValueError('GAST-Nav replay needs --resume <checkpoint>')
    saved = torch.load(args.resume, map_location=args.device, weights_only=False)
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    policy.eval()
    nav, manager = env.nav, env.reward_manager

    def step(obs):
        raw, reward, terminated, truncated, extras = env.step(policy.act_inference(obs))
        return TensorDict(raw, batch_size=[env.num_envs], device=args.device), reward, terminated | truncated

    with torch.inference_mode():
        for _ in range(warmup):
            obs, _, _ = step(obs)
        legacy_names = [n for n in manager.active_terms if n != 'nav_total']
        legacy_cols = [manager.active_terms.index(n) for n in legacy_names]
        nav_names = list(nav.reward_weighted)
        names = legacy_names + ['nav/' + n for n in nav_names]
        shim = _RateShim(names)
        real = env.reward_manager

        class NavLog(RolloutLog):
            def after_step(self, reward, done):
                rates = torch.cat((real._step_reward[:, legacy_cols],
                                   torch.stack([nav.reward_weighted_rm[n] for n in nav_names], -1)), -1)
                shim._step_reward = rates
                self.rewards = shim
                for key, value in (('nav_mode', nav.mode), ('nav_go', nav.go), ('nav_controller', nav.controller),
                                   ('need_stop', nav.last_need_stop), ('stopped', nav.last_stopped),
                                   ('stair_end_wait', nav.stair_end184.wait),
                                   ('task', torch.tensor([TASKS.index(t) if t in TASKS else 0 for t in nav.control.commands.task],
                                                         device=reward.device))):
                    self._add(key, value.float())
                super().after_step(reward, done)

            def finish(self, bootstrap, out):
                self.rec = {k: [v[:keep] for v in rows] for k, rows in self.rec.items()}
                self.n = min(keep, self.n)
                self.meta['num_envs'] = self.n
                return super().finish(bootstrap[:keep], out)

        env.reward_manager = shim  # only for RolloutLog.__init__ (term names)
        log = NavLog(env, task='Gd-GastNavV1-Rbq10-Dreamwaq-v0', checkpoint=str(args.resume), gamma=0.99, lam=0.95,
                     steps=steps, terrain_source='gast_history',
                     extra_meta={'driver': 'gast_nav', 'nav_iteration': saved.get('iteration'), 'stage': saved.get('stage'),
                                 'task_names': list(TASKS), 'mode_names': ['LEGACY', 'NAV', 'TURN']})
        env.reward_manager = real
        sums = {k: torch.zeros(len(names), dtype=torch.float64, device=args.device) for k in ('stopped', 'moving')}
        counts = {'stopped': 0, 'moving': 0}
        by_task = {}
        while not log.full():
            mean = policy.act_inference(obs)
            log.before_step(obs, mean=mean, std=torch.zeros_like(mean), action=mean,
                            value=policy.evaluate(obs).reshape(-1), latent=policy.terrain_latent(obs),
                            command=env.command_manager.get_command('base_velocity'))
            obs, reward, done = step(obs)
            log.after_step(reward, done)
            rates = shim._step_reward.double()
            need, stopped = nav.last_need_stop.bool(), nav.last_stopped.bool()
            for key, mask in (('stopped', need & stopped), ('moving', need & ~stopped)):
                sums[key] += rates[mask].sum(0)
                counts[key] += int(mask.sum())
            for i in need.nonzero().flatten().tolist():
                t = nav.control.commands.task[i]
                row = by_task.setdefault(t, [0, 0])
                row[0] += 1
                row[1] += int(stopped[i])
        log.finish(policy.evaluate(obs).reshape(-1), str(out))
    means = {k: (sums[k] / max(counts[k], 1)).tolist() for k in sums}
    table = sorted(({'term': n, 'stopped': means['stopped'][j], 'moving': means['moving'][j],
                     'moving_minus_stopped': means['moving'][j] - means['stopped'][j]} for j, n in enumerate(names)),
                   key=lambda r: -r['moving_minus_stopped'])
    stats = {'checkpoint': str(args.resume), 'iteration': saved.get('iteration'), 'envs': env.num_envs, 'steps': steps,
             'need_stop_ticks': counts, 'stop_success_rate': counts['stopped'] / max(1, sum(counts.values())),
             'by_task': {t: {'ticks': a, 'stopped_fraction': b / max(a, 1)} for t, (a, b) in by_task.items()},
             'rate_per_term': table,
             'note': 'mean reward RATE (per second, weight applied) over need-stop ticks; positive moving_minus_stopped '
                     'means the term pays the policy more for still moving when it should stop'}
    Path(str(out) + '.stop_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    return {'status': 'completed', 'replay': str(out), 'stop_stats': str(out) + '.stop_stats.json'}
