from unittest.mock import patch
import torch
from gd_lab.gast_nav.scene_types import SceneIndex
from gd_lab.gast_nav.terrains import build_scene
from gd_lab.gast_nav.scene_adapter import EpisodeCylinder,bind_scene
from gd_lab.gast_nav.moving_cylinder import NoCylinderPathError
from gd_lab.gast_nav.runner_support import rank_env_count,TopK
from types import SimpleNamespace

def test_rebind_curriculum_preserves_tiles():
    a=build_scene('nav_corridor',0.,seed=1)[2]
    b=build_scene('nav_door',.8,seed=2)[2]
    terrain=SimpleNamespace(cfg=SimpleNamespace(terrain_generator=SimpleNamespace(nav_scene_grid={(0,0):a,(1,0):b})),
                            terrain_levels=torch.tensor([0,0]),terrain_types=torch.tensor([0,0]))
    index=bind_scene(terrain)
    terrain.terrain_levels[1]=1;bind_scene(terrain,index,[1])
    assert index.lookup([0,1])==[a,b]

def test_cylinder_rejection_skips_only_rejected_episode():
    meta=build_scene('nav_corridor',0.,seed=1)[2]
    cylinder=EpisodeCylinder(SceneIndex([meta,meta]),stage='B')
    original=super(EpisodeCylinder,cylinder).reset
    def reset(ids,**kw):
        if ids==[0]:raise NoCylinderPathError(0,meta,.3,1)
        return original(ids,**kw)
    with patch('gd_lab.gast_nav.scene_adapter.MovingCylinder.reset',side_effect=reset):
        cylinder.reset(stage='C')
    assert not cylinder.enabled[0] and cylinder.position[0,2]<-10
    assert cylinder.skip_report()[0]['cylinder_skipped']>=1
    assert cylinder.skip_report()[0]['attempts']==2

def test_exact_4096_partition():
    assert [rank_env_count(4096,3,r) for r in range(3)]==[1366,1365,1365]

def test_top5_retains_best_and_discards_nan(tmp_path):
    top=TopK(tmp_path,5)
    for i in range(8):top.consider(float(i),i,lambda path:path.write_text('checkpoint'))
    assert [x[0] for x in top.records]==[7.,6.,5.,4.,3.]
    assert len(list(tmp_path.glob('*.pt')))==5
    assert not top.consider(float('nan'),9,lambda path:None)

def test_pilot_scope_guards(monkeypatch):
    from gd_lab.gast_nav.pilot_runner import validate_pilot
    import yaml
    from pathlib import Path
    cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/gast_nav/pilot.yaml').read_text())
    args=SimpleNamespace(pilot=True,distributed=True,stage='A0',total_envs=4096,max_iterations=200,pilot_smoke=False)
    monkeypatch.setenv('GAST_NAV_PILOT','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2')
    validate_pilot(args,cfg,3)
    import pytest
    with pytest.raises(ValueError):validate_pilot(args,cfg,4)
    args.max_iterations=201
    with pytest.raises(ValueError):validate_pilot(args,cfg,3)
    args.max_iterations=200;monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    with pytest.raises(ValueError):validate_pilot(args,cfg,3)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2');args.pilot_smoke=True;args.max_iterations=5;args.total_envs=48
    validate_pilot(args,cfg,3)
