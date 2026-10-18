"""Run explicitly on GPU0; CPU suite skips when CUDA visibility is disabled."""
import pytest,torch
from gd_lab.gast_nav.map_types import Grid,RayBatch
from gd_lab.gast_nav.ray_geometry import resolve_first_hits
from gd_lab.gast_nav.obstacle_map import ray_evidence_reference
from gd_lab.gast_nav.map_cuda import ray_evidence_cuda
from gd_lab.gast_nav.map_config import load_config
from gd_lab.gast_nav.near_clearance import clearance_reference
from gd_lab.gast_nav.map_cuda import clearance_cuda

@pytest.mark.skipif(not torch.cuda.is_available(),reason='explicit GPU0 equivalence test')
@pytest.mark.parametrize('seed',range(6))
def test_random_cuda_reference_bits(seed):
    g=torch.Generator().manual_seed(seed);n=3;r=97;grid=Grid(17,13,.1,-.8,-.6)
    origins=torch.rand(n,r,3,generator=g)*2-1;directions=torch.randn(n,r,3,generator=g)
    directions[:,::7,seed%3]=0
    distance=torch.rand(n,r,generator=g)*4
    rays=RayBatch(origins,directions,distance,torch.rand(n,r,generator=g)<.2)
    resolved=resolve_first_hits(rays)
    h=torch.rand(n,grid.ny,grid.nx,generator=g)*.5-.25;h[:,::7,::3]=torch.nan
    expected=ray_evidence_reference(resolved,h,grid=grid)
    from gd_lab.gast_nav.map_types import ResolvedRays
    gpu=ResolvedRays(*(getattr(resolved,k).cuda() for k in ('origins','directions','distances','occupied_hit','self_hit')))
    actual=ray_evidence_cuda(gpu,h.cuda(),grid,load_config())
    assert torch.equal(actual[0].cpu(),expected[0])
    assert all(torch.equal(a.cpu(),b) for a,b in zip(actual[1],expected[1]))
    state=torch.randint(0,3,(n,3,grid.ny,grid.nx),generator=g,dtype=torch.uint8)
    assert torch.equal(clearance_cuda(state.cuda(),grid).cpu(),clearance_reference(state,grid=grid))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='explicit GPU0 equivalence test')
def test_cuda_surface_covariance_tolerance():
    from gd_lab.gast_nav.map_cuda import fit_surface_cuda
    from gd_lab.gast_nav.obs_terms import fit_observed_surface
    from gd_lab.gast_nav.support import grow_support
    import yaml
    from pathlib import Path
    settings=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/gast_nav/environment.yaml').read_text())['sensors']
    g=torch.Generator().manual_seed(162);grid=Grid(17,13,.1,-.8,-.6)
    xy=torch.rand(3,500,2,generator=g)*2-1
    z=.2*xy[...,0]+.1*xy[...,1]+torch.randn(3,500,generator=g)*.001
    points=torch.cat((xy,z[...,None]),-1);valid=torch.rand(3,500,generator=g)>.1
    expected=fit_observed_surface(points,valid,grid,settings)
    actual=fit_surface_cuda(points.cuda(),valid.cuda(),grid,settings)
    assert torch.equal(actual[1].cpu(),expected[1])
    torch.testing.assert_close(actual[0].cpu(),expected[0],rtol=0,atol=0,equal_nan=True)
    # Reduction ordering can rotate fitted normals slightly; no h/mask tolerance.
    torch.testing.assert_close(actual[2].cpu(),expected[2],rtol=0,atol=1e-4,equal_nan=True)
    seeds=expected[1].clone()
    ref=grow_support(*expected,seeds,segment_type='STAIR_UP')
    got=grow_support(*actual,seeds.cuda(),segment_type='STAIR_UP')
    assert torch.equal(ref[1],got[1].cpu())
    assert torch.equal(ref[2],got[2].cpu())
