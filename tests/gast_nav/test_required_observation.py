import pytest
import torch
from gd_lab.gast_nav.supervisor import required_observed, required_band_indices
from gd_lab.gast_nav import schema as S


def test_b3_exclusion_preserves_map_and_lower_band_counterexamples():
    # B3 unknown no longer blocks, but missing B1/B2 or support still blocks.
    state = torch.tensor([[[[S.FREE, S.UNKNOWN, S.FREE, S.OCCUPIED, S.FREE]],
                           [[S.FREE, S.FREE, S.UNKNOWN, S.FREE, S.FREE]],
                           [[S.UNKNOWN] * 5]]], dtype=torch.uint8)
    original = state.clone()
    support = torch.tensor([[[True, True, True, True, False]]])
    assert required_observed(state, support, ['B1', 'B2']).tolist() == [[[True, False, False, True, False]]]
    assert not required_observed(state, support, ['B1', 'B2', 'B3']).any()
    assert torch.equal(state, original)
    # Thin pillar remains occupied. Wall-hidden lower-band unknown is not freed.
    assert state[0, 0, 0, 3] == S.OCCUPIED
    assert state[0, 0, 0, 1] == S.UNKNOWN


@pytest.mark.parametrize('bands', [[], ['B4'], ['B1', 'B1']])
def test_required_bands_fail_closed_config(bands):
    with pytest.raises(ValueError):
        required_band_indices(bands)


def test_missing_return_is_not_a_self_intersection():
    from gd_lab.gast_nav.map_types import RayBatch, DynamicCapsules
    from gd_lab.gast_nav.obs_terms import mask_self_returns
    from gd_lab.gast_nav.ray_geometry import resolve_first_hits
    rays = RayBatch(torch.tensor([[[0., 0., 0.]]]), torch.tensor([[[1., 0., 0.]]]),
                    torch.tensor([[float('inf')]]), torch.tensor([[False]]))
    capsule = DynamicCapsules(torch.tensor([[[0., 2., 0.]]]),
                              torch.tensor([[[0., 2., 1.]]]), torch.tensor([[.1]]))
    masked = mask_self_returns(rays, capsule)
    assert not masked.self_hit.any()
    assert torch.isinf(masked.distances).all()
    before, after = resolve_first_hits(rays), resolve_first_hits(masked)
    assert torch.equal(before.distances, after.distances)
    assert torch.equal(before.occupied_hit, after.occupied_hit)


def test_near_certificate_sidecar_preserves_all_band_policy_input():
    from test_maps import full_rays
    from gd_lab.gast_nav.near_clearance import build_near, extended_near_grid
    grid, _ = extended_near_grid()
    support = torch.zeros(1, grid.ny, grid.nx)
    original = build_near(full_rays(), support)
    frame, state, known = build_near(full_rays(), support, return_evidence=True)
    assert torch.equal(original.clearance, frame.clearance)
    assert torch.equal(original.unknown, frame.unknown)
    assert torch.equal(original.age_s, frame.age_s)
    assert torch.equal(frame.unknown, (state == S.UNKNOWN).any(1))
    assert known.all()
