import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
import torch
from gd_lab.gast_nav import schema as S
from gd_lab.gast_nav.map_types import RayBatch, MapFrame
from gd_lab.gast_nav.obstacle_map import ObstacleMap
from gd_lab.gast_nav.near_clearance import build_near, extended_near_grid
from gd_lab.gast_nav.packets import encode_obs1, decode_obs1, age_codes


@pytest.mark.parametrize('path', sorted(Path(__file__).with_name('golden').glob('obs1_*.json')), ids=lambda p: p.stem)
def test_scalar_golden_end_to_end(path):
    case = json.loads(path.read_text()); data, expected = case['input'], case['expected']
    rays = RayBatch(torch.tensor([data['ray_origins']]), torch.tensor([data['ray_directions']]),
                    torch.tensor([data['first_hit_distances']]), torch.tensor([data['self_hit']]))
    end = rays.origins+rays.directions*rays.distances[..., None]
    assert torch.allclose(end, torch.tensor([data['point_cloud']]))
    support = torch.full((1, S.WIDE_NY, S.WIDE_NX), data['support_height_default_m'])
    wide = ObstacleMap(1).update(rays, support, torch.tensor([data['sample_pose']], dtype=torch.float64),
                               torch.tensor([data['sample_time_ms']]))
    age = wide.age_s.clone(); age[0, 0, 21, 24] = data['wide_hit_age_s']; wide = replace(wide, age_s=age)
    grid, _ = extended_near_grid()
    near = build_near(rays, torch.full((1, grid.ny, grid.nx), data['support_height_default_m']))
    packet = encode_obs1(wide, near)
    reference = base64.b64decode(expected['packet_bytes_base64'])
    assert hashlib.sha256(reference).hexdigest() == expected['packet_sha256']
    assert packet == reference
    decoded = decode_obs1(reference)
    nonzero = decoded.wide.state[0].nonzero().tolist()
    values = [row+[int(decoded.wide.state[(0, *row)])] for row in nonzero]
    assert values == expected['wide_state_nonzero']
    assert bool((decoded.wide.support == expected['support_default_code']).all())
    assert decoded.near.clearance[0, 0].flatten().tolist() == expected['near_clearance_low']
    assert bool((decoded.near.clearance[0, 1] == expected['near_clearance_high_default']).all())
    assert bool(decoded.near.unknown.all())
