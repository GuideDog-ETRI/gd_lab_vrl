"""Shared UPDATE184 diagnostic and authorized local training fixture."""
from pathlib import Path
from types import SimpleNamespace
import json,os
ROOT=Path(__file__).resolve().parents[3]

def configure184(cfg,out,*,envs,seed=169,memory_ttl=.5):
    import torch,yaml
    from .env_cfg import GastNavEnvCfg
    a=SimpleNamespace(envs=envs,seed=seed,memory_ttl=memory_ttl,easy_c=True,pending167=True,measured167=True,envelope174=None,dense=False,rule='off',cert_ttl=.5,depth_scale=1,legacy_check=False)
    result={}
    cfg.seed=a.seed;torch.manual_seed(a.seed);cfg.scene.num_envs=a.envs;cfg.sim.device='cuda:0'
    cfg.scene.terrain.terrain_generator.num_rows=2;cfg.scene.terrain.terrain_generator.num_cols=20;cfg.scene.terrain.max_init_terrain_level=0
    if a.easy_c:
        import yaml
        from gd_lab.gast_nav.scene_config import load_scene_config
        from gd_lab.gast_nav.curriculum167 import easy_scene_settings
        limits=yaml.safe_load((ROOT/'configs/gast_nav/curriculum_C.yaml').read_text())['scene_limits']
        scene_file=out/'easy_scenes.yaml';scene_file.write_text(yaml.safe_dump(easy_scene_settings(load_scene_config(),limits)))
        for kind,terrain in cfg.scene.terrain.terrain_generator.sub_terrains.items():
            if kind.startswith('nav_'):terrain.settings_path=str(scene_file)
        cfg.nav_settings['curriculum_C_limits']=limits
        cfg.scene.terrain.terrain_generator.difficulty_range=(0.,0.)
        result['curriculum_C_limits']=limits
    if a.pending167:
        import yaml
        c=yaml.safe_load((ROOT/'configs/gast_nav/curriculum_C.yaml').read_text())
        cfg.nav_settings['rewards'].update(c['reward_overrides'])
        cfg.nav_settings['free_evidence_backend']='pending167'
        cfg.nav_settings['diagnostics167']=True
        cfg.nav_settings['sensors']['lower_depth']['diagnostic_unoccluded']=True
        result['free_evidence_backend']='pending167'
    if a.measured167:
        assert a.pending167, 'use the same C rewards/diagnostic fixture'
        cfg.nav_settings['free_evidence_backend']='coverage174'
        cfg.nav_settings['supervisor176']=yaml.safe_load((ROOT/'configs/gast_nav/supervisor177.yaml').read_text())
        cfg.nav_settings['sensors']['lower_depth']['enabled']=False
        result['free_evidence_backend']='coverage174'
        result['producer_status']='coverage174 + temporal169; no D4xx, no aperture, no mesh'
        cfg.nav_settings['temporal169_config']='configs/gast_nav/temporal169_training_c.yaml'
        result['seed']=a.seed
        result['volume_valid_producer']='coverage174 scene-only first-hit; no D4xx support'
    # UPDATE167 successful-subset 2D envelope is not a qualified 3D future trajectory.
    cfg.nav_settings.pop('swept_envelope_file',None)
    if a.envelope174:cfg.nav_settings['diagnostic_envelope174']=str(a.envelope174.resolve())
    if a.dense:
        from gd_lab.gast_nav.obs_terms import isaac_sensor_cfg
        cfg.nav_settings['sensors'].update(azimuth_count=48,elevation_count=32)
        cfg.scene.nav_obstacles=isaac_sensor_cfg(cfg.nav_settings['sensors'],prim_path='{ENV_REGEX_NS}/Robot/trunk',mesh_prim_path='/World/ground',update_period=cfg.nav_settings['capture_period_s'])
    cfg.nav_settings['observability']['r1_static_history']=a.rule in ('r1','both')
    cfg.nav_settings['observability']['r2_unknown_components']=a.rule in ('r2','both')
    cfg.nav_settings['observability']['certification_ttl_s']=a.cert_ttl
    result['certification_settings']=dict(cfg.nav_settings['observability'])
    cfg.nav_settings['sensors']['lower_depth']['sampling_scale']=a.depth_scale
    result['sensor_settings']=cfg.nav_settings['sensors'];result['simulation_turn_cap']=cfg.nav_settings['commands']['simulation_turn_cap_radps']
    if a.legacy_check:
        from dataclasses import replace
        import gd_lab.gast_nav.fixture174 as fixtures
        original_fixture=fixtures.fixture174
        def legacy_fixture(kind,settings_path,seed=174):
            f=original_fixture('nav_corridor',settings_path,seed)
            return replace(f,meta=replace(f.meta,kind='legacy'))
        fixtures.fixture174=legacy_fixture
    from gd_lab.gast_nav.fixture174 import configure_fixture174
    configure_fixture174(cfg.scene.terrain,a.envs,scene_file,seed=a.seed)
    cfg.scene.terrain.terrain_generator.coverage174_mixed=True
    from isaaclab.sensors import ContactSensorCfg
    import math
    cfg.scene.lazy_sensor_update=False
    for foot in ('FL_foot','FR_foot','RL_foot','RR_foot'):
        setattr(cfg.scene,'ground_contact174_'+foot,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+foot,filter_prim_paths_expr=[f'/World/ground/solids/box_{i}' for i in range(cfg.scene.terrain.terrain_generator.expected_contact_boxes174)],track_contact_points=True,history_length=math.ceil(.05/cfg.sim.dt)+1,max_contact_data_count_per_prim=32,update_period=cfg.sim.dt))
    for name in list(vars(cfg.curriculum)):
        if not name.startswith('_'):setattr(cfg.curriculum,name,None)
    # Primitive-only flat C has no legacy gap/stair rewards or metadata.
    # Retain the requested C nav reward, not fabricated legacy geometry.
    for name in list(vars(cfg.rewards)):
        if not name.startswith('_') and name!='nav_total':setattr(cfg.rewards,name,None)
    cfg.nav_stage_settings['moving_cylinder']=False
    cfg.scene.nav_cylinder.spawn.collision_props.collision_enabled=False
    cfg.nav_settings['primitive_c169']=True
    assert not cfg.nav_settings['observability']['r1_static_history'] and not cfg.nav_settings['observability']['r2_unknown_components']
    (out/'resolved_config_pre177.json').write_text(json.dumps(cfg.nav_settings,indent=2))
    from gd_lab.gast_nav.e1_terrain177 import E1Terrain177
    from isaaclab.managers import ObservationTermCfg
    cfg.observations.terrain.camera_visible=ObservationTermCfg(func=E1Terrain177)
    cfg.nav_settings['terrain_fusion177']=True
    cfg.nav_settings['pipeline179']=True
    cfg.nav_settings['pipeline180']=True
    cfg.nav_settings['current_body_margin_m']=yaml.safe_load((ROOT/'configs/gast_nav/pipeline179.yaml').read_text())['current_body_margin_m']
    cfg.nav_settings['front_depth183']=yaml.safe_load((ROOT/'configs/gast_nav/front_depth183.yaml').read_text())
    cfg.nav_settings['front_depth183'].update(enabled=True,obstacle_ttl_s=a.memory_ttl,support_ttl_s=a.memory_ttl)
    cfg.nav_settings['supervisor176'].update(obstacle_ttl_s=a.memory_ttl,support_ttl_s=a.memory_ttl)
    cfg.nav_settings['supervisor_mode']='shadow'
    return cfg
