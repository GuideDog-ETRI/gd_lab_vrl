"""Only simulator runner CONFIG is stubbed. Real 3906/torch/rsl_rl network code is used."""
import sys
import types
from pathlib import Path
import torch
import yaml
from tensordict import TensorDict
config = types.ModuleType('gd_lab.teachers.cvtt.agent_cfg')
config.DreamwaqVrlRunnerCfg = type('DreamwaqVrlRunnerCfg', (), {})
sys.modules['gd_lab.teachers.cvtt.agent_cfg'] = config
from gd_lab.gast.teacher import GastRayIrActorCritic
from gd_lab.gast_nav.actor_critic import GastNavActorCritic, NAV_FEATURE_DIM
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
SHA = 'f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98'

def runtime():
    return yaml.safe_load((ROOT / 'configs/gast_nav/runtime.yaml').read_text())

def observations(n=3):
    history = torch.randn(n, 8, 562)
    history[:, :, 187:374] = 1
    history[:, :, -1] = torch.arange(8) / 10
    return TensorDict({'policy': torch.randn(n, 230), 'critic': torch.randn(n, 298),
        'terrain': torch.cat((torch.randn(n, 187), torch.ones(n, 187)), -1),
        'gast_history': history.flatten(1), 'gast_clean': torch.zeros(n, 748),
        'nav_wide': torch.rand(n, 27, 43, 49), 'nav_near': torch.rand(n, 3, 25, 37),
        'nav_features': torch.rand(n, NAV_FEATURE_DIM), 'nav_privileged': torch.rand(n, 8)}, batch_size=[n])

def policy_pair(obs):
    args = dict(obs=obs, obs_groups={'policy':['policy'], 'critic':['critic']}, num_actions=12,
        actor_obs_normalization=True, critic_obs_normalization=True, actor_hidden_dims=[512,256,128],
        critic_hidden_dims=[512,256,128], activation='elu', history_length=5,
        policy_term_dims=[3,3,3,12,12,12,1], velocity_target_slice=(45,48), actor_history_steps=4,
        height_scan_start=110, adaboot_enabled=False)
    return GastRayIrActorCritic(**args), GastNavActorCritic(**args, nav_privileged_dim=8)
