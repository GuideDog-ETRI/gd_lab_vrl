"""Resolve local files and explicit oracle caps; never discovers live ROS state."""
import json
from pathlib import Path
import yaml
from gd_lab.gast_nav.temporal_map.manifest import effective_manifest
ROOT=Path(__file__).resolve().parents[2];BASE=Path('/home/user/gd_project');gdm=BASE/'gdm_docker/workspace/src'
s=yaml.safe_load((ROOT/'configs/gast_nav/temporal_map.yaml').read_text())
s['extrinsics_file']=str(BASE/'sim_docker/_sensor_configuration.yaml')
s['mid360_pattern_file']=str(BASE/'sim_docker/workspace/scripts/mid360_pattern.npy')
s['ouster_emitter_file']='/usr/local/lib/python3.11/dist-packages/isaacsim/exts/isaacsim.sensors.rtx/data/lidar_configs/Ouster/OS1/OS1_REV6_128ch10hz1024res.json'
s['gdm']={
 'planning_launch':str(gdm/'gdm_planning/launch/rbq.launch.py'),
 'sensor_configuration':str(gdm/'gdm_planning/resource/sensor_configuration.yaml'),
 'trav_rbq':str(gdm/'trav_map/trav_map/config/setups/rbq/rbq.yaml'),
 'trav_core':str(gdm/'trav_map/trav_map/config/core/core_param.yaml')}
out=ROOT/'logs/gast/nav_v1/update177'
(out/'temporal_runtime_settings.yaml').write_text(yaml.safe_dump(s))
manifest=effective_manifest(s)
(out/'temporal_effective_manifest.json').write_text(json.dumps(manifest,indent=2))
print(manifest['sha256'])
