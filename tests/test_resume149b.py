from types import SimpleNamespace
import hashlib
import torch
import pytest
from test_rayir_warm149 import Policy
from test_online_rollout_clean import _fake_env
from gd_lab.gast.resume149 import validate_resume, verify_loaded, reset_restored_env
from gd_lab.gast.warm_start import build_grouped_optimizer
from gd_lab.rl.online_rollout import install_episode_collector


def test_strict_optimizer_roundtrip_and_mismatch(tmp_path):
    p=Policy();o=build_grouped_optimizer(p,1e-4,1e-3,torch.optim.Adam(p.parameters()))
    sum(v.sum() for v in p.parameters()).backward();o.step();p.cenet.optimizer.step()
    task='Gd-GastRayIrV21-Rbq10-Dreamwaq-v0'
    c=dict(model_state_dict=p.state_dict(),optimizer_state_dict=o.state_dict(),iter=10,
        infos={'gd_lab':dict(gast_warm_start={'task':task},total_timesteps=1100*1024,total_envs=1024,
        terrain_level_means_by_family={'platform_gap':9.2},cenet_optimizer_state_dict=p.cenet.optimizer.state_dict())})
    path=tmp_path/'source.pt';torch.save(c,path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_resume(path,sha,task)['absolute_step_offset']==1100
    target=Policy();target.load_state_dict(c['model_state_dict']);target.cenet.optimizer.load_state_dict(c['infos']['gd_lab']['cenet_optimizer_state_dict'])
    opt=build_grouped_optimizer(target,1e-4,1e-3,torch.optim.Adam(target.parameters()));opt.load_state_dict(c['optimizer_state_dict'])
    alg=SimpleNamespace(policy=target,optimizer=opt)
    assert verify_loaded(alg,path)==dict(model_exact=True,ppo_exact=True,cenet_exact=True)
    opt.param_groups[0]['lr']=2e-4
    with pytest.raises(ValueError,match='value differs'):verify_loaded(alg,path)
    c.pop('optimizer_state_dict');torch.save(c,path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='missing optimizer'):validate_resume(path,sha,task)


def test_reset_preserves_levels_and_restores_curriculum_function():
    levels=torch.tensor([8,9]);origins=torch.zeros(2,3)
    compute=lambda: levels.add_(1)
    raw=SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_levels=levels),env_origins=origins),
                        curriculum_manager=SimpleNamespace(compute=compute))
    env=SimpleNamespace(unwrapped=raw,reset=lambda:raw.curriculum_manager.compute())
    reset_restored_env(env)
    assert levels.tolist()==[8,9]
    assert raw.curriculum_manager.compute is compute
    raw.curriculum_manager.compute();assert levels.tolist()==[9,10]


def test_top5_excludes_episodes_started_in_ramp():
    env=_fake_env(None);raw=env.unwrapped
    raw._aug149_rank_after_step=50000
    records=install_episode_collector(env,{'platform_gap':[0]})
    raw.common_step_counter=50099
    raw._reset_idx(torch.tensor([0,1]));assert records==[]
    raw.common_step_counter=50100
    raw._reset_idx(torch.tensor([0,1]));assert len(records)==2
