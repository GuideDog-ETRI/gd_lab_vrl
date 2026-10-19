"""Native 3906 subclass; decoded map tensors follow schema.py, no simulator state read."""
import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.networks import EmpiricalNormalization
from gd_lab.gast.teacher import GastRayIrActorCritic
from gd_lab.rl.distributed_sync import synchronize_empirical_normalizer
from . import schema

# mode(3), effective_go, v_cap, effective_turn_rate, phase(6), dwell progress, c(3).
NAV_FEATURE_DIM = len(schema.MODES) + 3 + len(schema.PHASES) + 1 + len(schema.CONDITION_KEYS)

class AdditiveLinear(nn.Module):
    def __init__(self, old, new_dim):
        super().__init__()
        self.weight = old.weight
        self.bias = old.bias
        self.old_dim = old.in_features
        self.W_new = nn.Parameter(old.weight.new_zeros(old.out_features, new_dim))

    def forward(self, x):
        return F.linear(x[..., :self.old_dim], self.weight, self.bias) + F.linear(x[..., self.old_dim:], self.W_new)

class MapCNN(nn.Sequential):
    def __init__(self, channels, output):
        super().__init__(nn.Conv2d(channels, 16, 3, padding=1), nn.ELU(),
                         nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ELU(),
                         nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(144, output), nn.ELU())

class GastNavActorCritic(GastRayIrActorCritic):
    """Extra obs keys: nav_wide[N,27,43,49], nav_near[N,3,25,37],
    nav_features[N,16], nav_route_features[N,3], nav_privileged[N,nav_privileged_dim]. Critic encoders are independent.
    Privileged layout is supplied explicitly by the environment, never guessed here.
    """
    def __init__(self, *args, nav_privileged_dim, **kwargs):
        super().__init__(*args, **kwargs)
        if nav_privileged_dim < 1:
            raise ValueError('declare the privileged observation width')
        self.nav_privileged_dim = nav_privileged_dim
        self.nav_wide_encoder = MapCNN(schema.WIDE_CHANNELS, 32)
        self.nav_near_encoder = MapCNN(schema.NEAR_CHANNELS, 16)
        self.nav_normalizer = EmpiricalNormalization(NAV_FEATURE_DIM)
        self.nav_critic_wide = MapCNN(schema.WIDE_CHANNELS, 32)
        self.nav_critic_near = MapCNN(schema.NEAR_CHANNELS, 16)
        self.nav_critic_normalizer = EmpiricalNormalization(NAV_FEATURE_DIM + nav_privileged_dim)
        self.actor[0] = AdditiveLinear(self.actor[0], 48 + NAV_FEATURE_DIM + schema.ROUTE_FEATURE_DIM)
        self.critic[0] = AdditiveLinear(self.critic[0], 48 + NAV_FEATURE_DIM + nav_privileged_dim + schema.ROUTE_FEATURE_DIM)
        self.nav_route_normalizer = EmpiricalNormalization(schema.ROUTE_FEATURE_DIM)
        self.nav_critic_route_normalizer = EmpiricalNormalization(schema.ROUTE_FEATURE_DIM)
        self.nav_update_old_stats = False
        self.nav_update_new_stats = True
        self.nav_cenet_step = False
        self.nav_aux_step = False

    def _maps(self, obs, critic=False):
        wide, near = obs['nav_wide'], obs['nav_near']
        if wide.shape[1:] != (schema.WIDE_CHANNELS, schema.WIDE_NY, schema.WIDE_NX):
            raise ValueError('wide shape violates schema')
        if near.shape[1:] != (schema.NEAR_CHANNELS, schema.NEAR_NY, schema.NEAR_NX):
            raise ValueError('near shape violates schema')
        a, b = (self.nav_critic_wide, self.nav_critic_near) if critic else (self.nav_wide_encoder, self.nav_near_encoder)
        return a(wide), b(near)

    def _route(self, obs):
        route = obs[schema.ROUTE_OBS_KEY]
        if route.shape != (obs['nav_features'].shape[0], schema.ROUTE_FEATURE_DIM):
            raise ValueError('route observation violates UPDATE179 schema')
        return route

    def _actor_input(self, obs, *, inference):
        old = super()._actor_input(obs, inference=inference)
        wide, near = self._maps(obs)
        return torch.cat((old, wide, near, self.nav_normalizer(obs['nav_features']),
                          self.nav_route_normalizer(self._route(obs))), -1)

    def evaluate(self, obs, **kwargs):
        old = self.critic_obs_normalizer(self.get_critic_obs(obs))
        wide, near = self._maps(obs, critic=True)
        extra = self.nav_critic_normalizer(torch.cat((obs['nav_features'], obs['nav_privileged']), -1))
        return self.critic(torch.cat((old, wide, near, extra,
                                            self.nav_critic_route_normalizer(self._route(obs))), -1))

    def update_normalization(self, obs):
        if self.nav_update_old_stats:
            super().update_normalization(obs)
        if self.nav_update_new_stats:
            synchronize_empirical_normalizer(self.nav_normalizer, obs['nav_features'])
            synchronize_empirical_normalizer(self.nav_route_normalizer, self._route(obs))
            synchronize_empirical_normalizer(self.nav_critic_route_normalizer, self._route(obs))
            synchronize_empirical_normalizer(self.nav_critic_normalizer,
                                             torch.cat((obs['nav_features'], obs['nav_privileged']), -1))
