"""GPU equivalence adapter: run unchanged CPU golden fixtures through CUDA kernels."""
import torch
from gd_lab.gast_nav import obstacle_map,near_clearance
from gd_lab.gast_nav.map_types import ResolvedRays

def pytest_configure(config):
    original=obstacle_map.ray_evidence
    def cuda_evidence(resolved,support_height,**kw):
        gpu=ResolvedRays(*(getattr(resolved,k).cuda() for k in ('origins','directions','distances','occupied_hit','self_hit')))
        actual=original(gpu,support_height.cuda(),**kw)
        expected=obstacle_map.ray_evidence_reference(resolved,support_height,**kw)
        actual=(actual[0].cpu(),tuple(t.cpu() for t in actual[1]))
        assert torch.equal(actual[0],expected[0]),'CUDA evidence state differs from golden'
        assert all(torch.equal(a,b) for a,b in zip(actual[1],expected[1])),'CUDA sub-bin evidence differs from golden'
        return actual
    obstacle_map.ray_evidence=cuda_evidence;near_clearance.ray_evidence=cuda_evidence
    clearance=near_clearance.clearance_from_state
    def cuda_clearance(state,**kw):
        actual=clearance(state.cuda(),**kw).cpu();expected=near_clearance.clearance_reference(state,**kw)
        assert torch.equal(actual,expected),'CUDA clearance differs from golden'
        return actual
    near_clearance.clearance_from_state=cuda_clearance

    from gd_lab.gast_nav import obs_terms
    fit=obs_terms.fit_observed_surface
    def cuda_fit(points,valid,grid,settings):
        actual=tuple(x.cpu() for x in fit(points.cuda(),valid.cuda(),grid,settings))
        expected=obs_terms.fit_observed_surface_reference(points,valid,grid,settings)
        torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0,equal_nan=True)
        assert torch.equal(actual[1],expected[1])
        torch.testing.assert_close(actual[2],expected[2],atol=1e-4,rtol=0,equal_nan=True)
        return actual
    obs_terms.fit_observed_surface=cuda_fit
