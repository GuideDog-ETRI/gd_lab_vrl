import numpy as np
import pytest
from gd_lab.gast_nav.v0_eval import paired_mesh
from gd_lab.gast_nav.terrains import KINDS

@pytest.mark.parametrize('kind',KINDS)
def test_structure_free_pair_preserves_floor_and_steps(kind):
    case=dict(kind=kind,seed=16200,structures=True,mode='gt')
    full,origin,meta=paired_mesh(case,.1)
    bare,origin2,meta2=paired_mesh(dict(case,structures=False),.1)
    assert np.array_equal(origin,origin2)
    assert meta.route_polyline==meta2.route_polyline
    assert all(b.kind=='step' for b in meta2.obstacles)
    removed=[b for b in meta.obstacles if b.kind!='step']
    expected=[m for m in full if not any(np.allclose(m.centroid,origin+np.array(b.center),atol=1e-8,rtol=0) for b in removed)]
    assert len(bare)==len(expected)<len(full)
    for a,b in zip(bare,expected):assert np.array_equal(a.vertices,b.vertices)
