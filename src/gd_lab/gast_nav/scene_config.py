"""Scene settings without Isaac imports or implicit filesystem writes."""
from pathlib import Path
import yaml


def load_scene_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[3]/'configs/gast_nav/scenes.yaml'
    value = yaml.safe_load(path.read_text())
    for key in ('terrain', 'corridor', 'door', 'turn_room', 'stairs', 'oracle', 'cylinder'):
        if key not in value: raise ValueError(f'missing scene config: {key}')
    if len(value['terrain']['size']) != 2 or min(value['terrain']['size']) <= 0 or value['terrain']['levels'] < 2:
        raise ValueError('invalid terrain settings')
    return value
