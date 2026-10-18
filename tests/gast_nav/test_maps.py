from dataclasses import replace
import math
import struct
import zlib
import pytest
import torch
from gd_lab.gast_nav import schema as S
from gd_lab.gast_nav.map_config import load_config
from gd_lab.gast_nav.map_types import (RayBatch, DynamicBoxes, DynamicCapsules, Grid, WIDE_GRID, MapFrame, NearFrame)
from gd_lab.gast_nav.support import encode_support, decode_support, grow_support, contact_seed_mask
from gd_lab.gast_nav.ray_geometry import resolve_first_hits, capsule_intersections
from gd_lab.gast_nav.obstacle_map import ray_evidence, subbin_edges, ObstacleMap
from gd_lab.gast_nav.near_clearance import extended_near_grid, build_near, clearance_from_state
from gd_lab.gast_nav.warp import warp_categorical
from gd_lab.gast_nav.packets import (age_codes, age_upper_s, pack_bits, unpack_bits, encode_obs1, decode_obs1, OBS1Receiver, HEADER_BYTES, PACKET_BYTES)
from gd_lab.gast_nav.map_inputs import network_inputs, packed_network_inputs


def rays_at(zs, *, n=1, dist=math.inf, self_hit=False):
    z = torch.tensor(zs, dtype=torch.float32)
    o = torch.zeros(n, len(z), 3); o[..., 0] = -.3; o[..., 2] = z
    d = torch.zeros_like(o); d[..., 0] = 1
    return RayBatch(o, d, torch.full((n, len(z)), dist), torch.full((n, len(z)), self_hit))


def full_rays(*, n=1):
    return rays_at([(a+b)/2 for band in subbin_edges() for a, b in band], n=n)


def blank_frames(n=1, *, stamp=1000, epoch=0, seq=0):
    state = torch.zeros(n, 3, S.WIDE_NY, S.WIDE_NX, dtype=torch.uint8)
    age = torch.full_like(state, torch.inf, dtype=torch.float32)
    support = torch.full((n, S.WIDE_NY, S.WIDE_NX), 50, dtype=torch.uint8)
    wide = MapFrame(state, age, support, torch.zeros(n, 4, dtype=torch.float64), torch.full((n,), stamp, dtype=torch.int64),
                    torch.full((n,), seq, dtype=torch.int64), torch.full((n,), epoch, dtype=torch.int64))
    near = NearFrame(torch.full((n, 2, S.NEAR_NY, S.NEAR_NX), 255, dtype=torch.uint8),
                     torch.ones(n, S.NEAR_NY, S.NEAR_NX, dtype=torch.bool), torch.full((n, S.NEAR_NY, S.NEAR_NX), torch.inf))
    return wide, near


@pytest.mark.parametrize('age,expected', [(.49, 5), (.50, 5), (.51, 6), (1.5, 15), (math.inf, 15), (0., 0)])
def test_age_ceil(age, expected):
    assert int(age_codes(torch.tensor([age]))[0]) == expected


@pytest.mark.parametrize('bits', [1, 2, 4])
def test_bit_packing(bits):
    x = torch.arange(13).to(torch.uint8) % (1 << bits)
    assert torch.equal(unpack_bits(pack_bits(x, bits), 13, bits), x)


def test_known_bits_lsb_and_padding():
    assert pack_bits(torch.tensor([0, 1, 2, 0], dtype=torch.uint8), 2).hex() == '24'
    assert pack_bits(torch.tensor([5, 15], dtype=torch.uint8), 4).hex() == 'f5'
    with pytest.raises(ValueError, match='padding'): unpack_bits(b'\xff', 1, 2)


def test_support_encoding():
    z = torch.tensor([[[-1., 0., 1.54, 4.08, torch.nan]]])
    c = encode_support(z)
    assert c.tolist() == [[[0, 50, 127, 254, 255]]]
    h, valid = decode_support(c)
    assert torch.allclose(h[valid], z[valid])
    assert not valid[0, 0, -1]


def grow(z, *, observed=None, normals=None, candidate=None, segment_type='FLAT'):
    z = torch.tensor([z], dtype=torch.float32)
    obs = torch.isfinite(z) if observed is None else observed
    normal = torch.zeros(*z.shape, 3); normal[..., 2] = 1
    if normals is not None: normal = normals
    seed = torch.zeros_like(obs); seed[:, :, 0] = True
    return grow_support(z, obs, normal, seed, segment_type=segment_type, current_support_height=torch.tensor([0.]), surface_candidate=candidate)


def test_box_04_is_not_support():
    _, _, valid = grow([[0., 0., .4, .4]])
    assert valid.tolist() == [[[True, True, False, False]]]


def test_stair_treads_are_support():
    _, _, valid = grow([[0., .2, .4, .6, .8]], segment_type='STAIR_UP')
    assert bool(valid.all())


def test_gap_is_unknown_not_obstacle():
    height, code, valid = grow([[0., 0., torch.nan, 0.]])
    assert code[0, 0, 2:].tolist() == [255, 255]
    assert not bool(valid[0, 0, 2:].any())


def test_support_requires_observed_horizontal_surface():
    normals = torch.zeros(1, 1, 3, 3); normals[..., 2] = 1; normals[0, 0, 1] = torch.tensor([1., 0., 0.])
    _, _, valid = grow([[0., .2, .4]], normals=normals, segment_type='STAIR_UP')
    assert valid.tolist() == [[[True, False, False]]]


def test_flat_seat_and_small_box_are_obstacles_not_support():
    reference, codes, valid = grow([[0., .2, .4, .45]])
    assert valid.tolist() == [[[True, False, False, False]]]
    assert reference.tolist() == [[[0., 0., 0., 0.]]]
    assert codes.tolist() == [[[50, 50, 50, 50]]]
    r = rays_at([.45], dist=.6)
    # Origin x=-.3 -> hit x=.3. Surface remains above current floor (B2).
    state, _ = ray_evidence(resolve_first_hits(r), reference, grid=Grid(4, 1, .1, 0., 0.))
    assert state[0, 1, 0, 3] == S.OCCUPIED
    reference2, _, is_support = grow([[0., .2]])
    assert not is_support[0, 0, 1]
    state, _ = ray_evidence(resolve_first_hits(rays_at([.2], dist=.4)), reference2, grid=Grid(2, 1, .1, 0., 0.))
    assert state[0, 0, 0, 1] == S.OCCUPIED


def test_no_hint_means_flat_and_measured_current_support_required():
    z = torch.zeros(1, 1, 1); normal = torch.tensor([[[[0., 0., 1.]]]])
    with pytest.raises(ValueError, match='current_support_height'):
        grow_support(z, torch.ones_like(z, dtype=torch.bool), normal, torch.ones_like(z, dtype=torch.bool))


def test_contact_seed_does_not_wrap_or_fill_missing_support():
    p = torch.tensor([[[0., 0., -.4], [100., 0., -.4], [torch.nan, 0., 0.]]])
    m = contact_seed_mask(p, torch.tensor([[True, True, True]]))
    assert int(m.sum()) == 1 and m[0, 21, 24]


def test_subbin_counts():
    assert [len(x) for x in subbin_edges()] == [3, 5, 14]


def test_all_subbins_free_but_thin_coverage_unknown():
    grid = Grid(1, 1, .1, 0., 0.)
    support = torch.zeros(1, 1, 1)
    state, diag = ray_evidence(resolve_first_hits(full_rays()), support, grid=grid)
    assert state.tolist() == [[[[1]], [[1]], [[1]]]]
    sparse, _ = ray_evidence(resolve_first_hits(rays_at([.15])), support, grid=grid)
    assert int(sparse[0, 0, 0, 0]) == S.UNKNOWN
    assert not bool(diag[0].logical_not().any())


def test_unknown_support_remains_unknown_with_hit():
    grid = Grid(1, 1, .1, 0., 0.)
    state, _ = ray_evidence(resolve_first_hits(rays_at([.2], dist=.3)), torch.full((1, 1, 1), torch.nan), grid=grid)
    assert bool((state == S.UNKNOWN).all())


def test_hit_occupied_beats_free():
    r = full_rays(); r = replace(r, distances=r.distances.clone()); r.distances[0, 0] = .3
    state, _ = ray_evidence(resolve_first_hits(r), torch.zeros(1, 1, 1), grid=Grid(1, 1, .1, 0., 0.))
    assert state[0, 0, 0, 0] == S.OCCUPIED


def test_self_hit_blocks_free_and_never_occupied():
    r = rays_at([.15], dist=.3, self_hit=True)
    resolved = resolve_first_hits(r)
    state, diag = ray_evidence(resolved, torch.zeros(1, 1, 3), grid=Grid(3, 1, .1, -.1, 0.))
    assert not bool((state == S.OCCUPIED).any())
    assert not bool(diag[0][0, 0, 2].any())


def test_wall_hides_dynamic_person():
    r = rays_at([.5], dist=.5)  # wall at x=.2
    cap = DynamicCapsules(torch.tensor([[[.8, 0., 0.]]]), torch.tensor([[[.8, 0., 1.8]]]), torch.tensor([[.2]]))
    resolved = resolve_first_hits(r, capsules=cap)
    assert float(resolved.distances[0, 0]) == .5
    state, _ = ray_evidence(resolved, torch.zeros(1, 1, 13), grid=Grid(13, 1, .1, -.3, 0.))
    assert state[0, 1, 0, 5] == S.OCCUPIED
    assert bool((state[0, :, :, 7:] == S.UNKNOWN).all())


def test_dynamic_box_can_precede_self_and_static():
    r = rays_at([.5], dist=1., self_hit=True)
    box = DynamicBoxes(torch.tensor([[[.3, 0., .5]]]), torch.tensor([[[.1, .2, .2]]]), torch.zeros(1, 1))
    resolved = resolve_first_hits(r, boxes=box)
    assert torch.allclose(resolved.distances, torch.tensor([[.5]]))
    assert bool(resolved.occupied_hit.all()) and not bool(resolved.self_hit.any())


def test_capsule_parallel_endcap_and_inside():
    cap = DynamicCapsules(torch.tensor([[[0., 0., 0.]]]), torch.tensor([[[0., 0., 1.]]]), torch.tensor([[.2]]))
    o = torch.tensor([[[0., 0., 2.], [0., 0., .5]]])
    d = torch.tensor([[[0., 0., -1.], [1., 0., 0.]]])
    hit = capsule_intersections(o, d, cap, eps=load_config().intersection_epsilon)
    assert torch.allclose(hit[..., 0], torch.tensor([[.8, 0.]]), atol=1e-6)


def test_env_isolation_and_ray_normalization():
    r = rays_at([.15], n=2, dist=.3)
    o = r.origins.clone(); o[1, :, 1] = 2.
    r = replace(r, origins=o, directions=r.directions*4)
    state, _ = ray_evidence(resolve_first_hits(r), torch.zeros(2, 1, 1), grid=Grid(1, 1, .1, 0., 0.))
    assert state[0, 0, 0, 0] == S.OCCUPIED
    assert bool((state[1] == S.UNKNOWN).all())


def test_clearance_cell_boundary_and_unknown_not_obstacle():
    grid = Grid(5, 1, .05, -.1, 0.)
    state = torch.zeros(1, 3, 1, 5, dtype=torch.uint8); state[0, 0, 0, 2] = S.OCCUPIED
    c = clearance_from_state(state, grid=grid)
    assert torch.allclose(c[0, 0, 0], torch.tensor([.075, .025, 0., .025, .075]), atol=1e-6)
    assert bool((c[0, 1] == .5).all())


def test_near_independent_raw_grid_and_unknown_or():
    grid, pad = extended_near_grid()
    assert (grid.nx, grid.ny, pad) == (57, 45, 10)
    near = build_near(rays_at([.15], dist=.3), torch.zeros(1, grid.ny, grid.nx))
    assert near.clearance.shape == (1, 2, 25, 37)
    assert near.clearance[0, 0, 12, 18] == 0
    assert near.clearance[0, 1, 12, 18] == 255
    assert near.unknown[0, 12, 18]  # B2/B3 not fully traversed


def test_near_roi_padding_includes_obstacle_just_outside_roi():
    grid, _ = extended_near_grid()
    near = build_near(rays_at([.15], dist=1.3), torch.zeros(1, grid.ny, grid.nx))  # x=1.0 outside +.9 ROI
    assert 0 < int(near.clearance[0, 0, 12, -1]) < 255


@pytest.mark.parametrize('yaw', [0., .1, math.pi/4, math.pi/2])
def test_warp_occupied_not_erased(yaw):
    grid = Grid(5, 5, .1, -.2, -.2)
    state = torch.zeros(1, 3, 5, 5, dtype=torch.uint8); state[0, 0, 2, 2] = S.OCCUPIED
    age = torch.zeros_like(state, dtype=torch.float32)
    out, _, _ = warp_categorical(state, age, torch.zeros(1, 4), torch.tensor([[0., 0., 0., yaw]]), grid=grid)
    assert bool((out == S.OCCUPIED).any())


def test_warp_identity_and_outside_unknown():
    grid = Grid(3, 3, .1, -.1, -.1)
    state = torch.ones(1, 3, 3, 3, dtype=torch.uint8); state[0, 0, 1, 1] = S.OCCUPIED
    age = torch.full_like(state, .2, dtype=torch.float32)
    out, a, _ = warp_categorical(state, age, torch.zeros(1, 4), torch.zeros(1, 4), grid=grid)
    assert torch.equal(out, state) and torch.equal(a, age)
    out, _, _ = warp_categorical(state, age, torch.zeros(1, 4), torch.tensor([[.1, 0., 0., 0.]]), grid=grid)
    assert bool((out[..., -1] == S.UNKNOWN).all())
    assert out[0, 0, 1, 0] == S.OCCUPIED


def test_warp_free_not_expanded_into_unknown():
    grid = Grid(3, 1, .1, -.1, 0.)
    state = torch.zeros(1, 3, 1, 3, dtype=torch.uint8); state[..., 1] = S.FREE
    out, _, _ = warp_categorical(state, torch.zeros_like(state, dtype=torch.float32), torch.zeros(1, 4), torch.tensor([[.05, 0., 0., 0.]]), grid=grid)
    assert not bool((out == S.FREE).any())


def test_obstacle_history_carry_ttl_and_reset():
    mapper = ObstacleMap(1)
    h = torch.zeros(1, S.WIDE_NY, S.WIDE_NX); pose = torch.zeros(1, 4)
    a = mapper.update(rays_at([.15], dist=.3), h, pose, torch.tensor([1000]))
    assert a.state[0, 0, 21, 24] == S.OCCUPIED
    b = mapper.update(rays_at([.15]), h, pose, torch.tensor([1490]))
    assert b.state[0, 0, 21, 24] == S.OCCUPIED
    assert float(b.age_s[0, 0, 21, 24]) == pytest.approx(.49)
    c = mapper.update(rays_at([.15]), h, pose, torch.tensor([1510]))
    assert c.state[0, 0, 21, 24] == S.UNKNOWN
    assert len(mapper.history) == 3
    assert torch.equal(a.support, mapper.history[2].support)
    mapper.reset()
    d = mapper.update(rays_at([.15]), h, pose, torch.tensor([100]))
    assert (int(d.epoch[0]), int(d.seq[0])) == (1, 0)
    assert not bool((d.state == S.OCCUPIED).any())
    assert a.state[0, 0, 21, 24] == S.OCCUPIED  # pinned original was not mutated


def test_partial_reset_preserves_other_env_epoch():
    mapper = ObstacleMap(2)
    h = torch.zeros(2, S.WIDE_NY, S.WIDE_NX); pose = torch.zeros(2, 4)
    mapper.update(rays_at([.15], n=2, dist=.3), h, pose, torch.tensor([1000, 1000]))
    mapper.reset([0])
    frame = mapper.update(rays_at([.15], n=2), h, pose, torch.tensor([1100, 1100]))
    assert frame.epoch.tolist() == [1, 0] and frame.seq.tolist() == [0, 1]
    assert frame.state[0, 0, 21, 24] != S.OCCUPIED and frame.state[1, 0, 21, 24] == S.OCCUPIED


@pytest.mark.parametrize('age,known', [(.49, True), (.50, True), (.51, False), (1.5, False)])
def test_packet_ttl_boundaries(age, known):
    wide, near = blank_frames(); wide.state[0, 0, 21, 24] = S.OCCUPIED; wide.age_s[0, 0, 21, 24] = age
    near.unknown[:] = False; near.age_s[:] = age
    decoded = decode_obs1(encode_obs1(wide, near))
    assert bool(decoded.wide.state[0, 0, 21, 24] == S.OCCUPIED) == known
    assert bool((~decoded.near.unknown).all()) == known


def test_transport_age_not_refreshed_by_receive():
    wide, near = blank_frames(); wide.state[0, 0, 21, 24] = S.OCCUPIED; wide.age_s[0, 0, 21, 24] = .49
    packet = encode_obs1(wide, near)
    assert decode_obs1(packet, now_ms=1001).wide.state[0, 0, 21, 24] == S.UNKNOWN


def test_packets_length_crc_reserved_and_metadata():
    wide, near = blank_frames(); packet = encode_obs1(wide, near)
    assert len(packet) == PACKET_BYTES
    with pytest.raises(ValueError, match='length'): decode_obs1(packet[:-1])
    corrupt = bytearray(packet); corrupt[-10] ^= 1
    with pytest.raises(ValueError, match='CRC'): decode_obs1(bytes(corrupt))
    reserved = bytearray(packet); reserved[HEADER_BYTES] |= 3
    reserved[-4:] = struct.pack('<I', zlib.crc32(reserved[:-4]))
    with pytest.raises(ValueError, match='reserved'): decode_obs1(bytes(reserved))
    meta = bytearray(packet); meta[68] ^= 1
    with pytest.raises(ValueError, match='metadata'): decode_obs1(bytes(meta))


def test_receiver_epoch_seq_and_future():
    wide, near = blank_frames(); packet = encode_obs1(wide, near); receiver = OBS1Receiver()
    receiver.receive(packet, now_ms=1000)
    with pytest.raises(ValueError, match='duplicate'): receiver.receive(packet, now_ms=1000)
    wide2 = replace(wide, epoch=torch.tensor([1]))
    receiver.receive(encode_obs1(wide2, near), now_ms=1000)
    with pytest.raises(ValueError, match='old/duplicate'): receiver.receive(packet, now_ms=1000)
    with pytest.raises(ValueError, match='future'): decode_obs1(packet, now_ms=999)


def test_network_shapes_channel_order_and_dense_packed_parity():
    wide, near = blank_frames(); wide.state[0, 0, 21, 24] = S.OCCUPIED; wide.age_s[0, 0, 21, 24] = .12
    wide.state[0, 1, 21, 24] = S.FREE; wide.age_s[0, 1, 21, 24] = .01
    near.unknown[:] = False; near.age_s[:] = 0.
    pose = torch.zeros(1, 4, dtype=torch.float64)
    a, b = network_inputs([wide], near, pose, 1010)
    c, d = packed_network_inputs([encode_obs1(wide, near)], pose, 1010)
    assert a.shape == (1, 27, 43, 49) and b.shape == (1, 3, 25, 37)
    assert torch.equal(a, c) and torch.equal(b, d)
    assert a[0, 0, 21, 24] == 1 and a[0, 4, 21, 24] == 1
    assert a[0, 2, 21, 24] == pytest.approx(.21/1.5)
    assert bool((a[:, 11::3] == 1).all())  # missing frame age=1


def test_network_old_epoch_unknown_and_sample_pose_warp_once():
    current, near = blank_frames(epoch=1, seq=0)
    old, _ = blank_frames(epoch=0, seq=100, stamp=900)
    old.state[:] = S.OCCUPIED; old.age_s[:] = 0.
    current.state[0, 0, 21, 24] = S.OCCUPIED; current.age_s[0, 0, 21, 24] = 0.
    pose = torch.tensor([[.1, 0., 0., 0.]], dtype=torch.float64)
    a, _ = network_inputs([current, old], near, pose, 1000)
    assert a[0, 0, 21, 23] == 1
    assert not bool(a[:, 9:11].any())


def test_invalid_rays_rejected():
    r = rays_at([.2]); d = r.directions.clone(); d[:] = 0.
    with pytest.raises(ValueError, match='invalid rays'): resolve_first_hits(replace(r, directions=d))


def test_near_age_and_pose_conversion():
    wide, near = blank_frames(); near.unknown[:] = False; near.age_s[:] = .5
    a, fine = network_inputs([wide], near, torch.zeros(1, 4, dtype=torch.float64), 1001)
    assert bool((fine[:, 2] == 1).all())
    near.age_s[:] = 0.; near.clearance[:, 0] = 0
    _, fine = network_inputs([wide], near, torch.tensor([[.05, 0., 0., 0.]], dtype=torch.float64), 1000)
    assert bool((fine[:, 0, :, :-1] == 0).all()) and bool((fine[:, 2, :, -1] == 1).all())


def test_subbin_boundary_ray_not_counted_twice():
    r = rays_at([.13, .23])
    state, passed = ray_evidence(resolve_first_hits(r), torch.zeros(1, 1, 1), grid=Grid(1, 1, .1, 0., 0.))
    assert not bool(passed[0].any())
    assert state[0, 0, 0, 0] == S.UNKNOWN


def test_wall_free_cannot_reach_occluded_person_subbins():
    r = full_rays(); r = replace(r, distances=torch.full_like(r.distances, .5))
    cap = DynamicCapsules(torch.tensor([[[.8, 0., 0.]]]), torch.tensor([[[.8, 0., 1.8]]]), torch.tensor([[.2]]))
    state, passed = ray_evidence(resolve_first_hits(r, capsules=cap), torch.zeros(1, 1, 13), grid=Grid(13, 1, .1, -.3, 0.))
    assert bool((state[..., 6:] == S.UNKNOWN).all())
    assert all(not bool(x[..., 6:, :].any()) for x in passed)


def test_rotated_dynamic_box_nearest_distance():
    r = rays_at([.5]); r = replace(r, origins=torch.tensor([[[-1., 0., .5]]]))
    box = DynamicBoxes(torch.tensor([[[0., 0., .5]]]), torch.tensor([[[.2, .1, .2]]]), torch.tensor([[math.pi/2]]))
    resolved = resolve_first_hits(r, boxes=box)
    assert float(resolved.distances[0, 0]) == pytest.approx(.9)


def test_box_inside_ray_starts_at_zero():
    r = rays_at([.5]); r = replace(r, origins=torch.tensor([[[0., 0., .5]]]))
    box = DynamicBoxes(torch.tensor([[[0., 0., .5]]]), torch.tensor([[[.2, .1, .2]]]), torch.tensor([[0.]]))
    assert resolve_first_hits(r, boxes=box).distances[0, 0] == 0


def test_capsule_zero_length_is_sphere():
    cap = DynamicCapsules(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), torch.tensor([[.2]]))
    o = torch.tensor([[[-1., 0., 0.]]]); d = torch.tensor([[[1., 0., 0.]]])
    assert float(capsule_intersections(o, d, cap, eps=load_config().intersection_epsilon)[0, 0, 0]) == pytest.approx(.8)


def test_static_return_farther_than_range_does_not_become_occupied():
    r = rays_at([.15], dist=100.)
    resolved = resolve_first_hits(r)
    assert not bool(resolved.occupied_hit.any())
    assert float(resolved.distances.max()) == load_config().ray_max_range_m


def test_carry_refresh_only_actual_new_hit_not_free_ray():
    mapper = ObstacleMap(1)
    h = torch.zeros(1, S.WIDE_NY, S.WIDE_NX); pose = torch.zeros(1, 4)
    mapper.update(rays_at([.15], dist=.3), h, pose, torch.tensor([1000]))
    frame = mapper.update(full_rays(), h, pose, torch.tensor([1400]))
    assert frame.state[0, 0, 21, 24] == S.OCCUPIED
    assert frame.age_s[0, 0, 21, 24] == pytest.approx(.4)
    frame = mapper.update(full_rays(), h, pose, torch.tensor([1600]))
    assert frame.state[0, 0, 21, 24] == S.FREE


def test_current_hit_refreshes_occupied_age():
    mapper = ObstacleMap(1)
    h = torch.zeros(1, S.WIDE_NY, S.WIDE_NX); pose = torch.zeros(1, 4)
    mapper.update(rays_at([.15], dist=.3), h, pose, torch.tensor([1000]))
    frame = mapper.update(rays_at([.15], dist=.3), h, pose, torch.tensor([1400]))
    assert frame.age_s[0, 0, 21, 24] == 0


def test_capture_timestamp_must_increase_within_epoch():
    mapper = ObstacleMap(1)
    h = torch.zeros(1, S.WIDE_NY, S.WIDE_NX); pose = torch.zeros(1, 4)
    mapper.update(rays_at([.15]), h, pose, torch.tensor([1000]))
    with pytest.raises(ValueError, match='timestamp'):
        mapper.update(rays_at([.15]), h, pose, torch.tensor([999]))


def test_network_dense_packed_with_expired_cells_and_invalid_support():
    wide, near = blank_frames()
    wide.state[:, 0] = S.OCCUPIED; wide.age_s[:, 0] = .51
    wide.state[:, 1] = S.FREE; wide.age_s[:, 1] = 0
    wide.support[0, 21, 24] = S.SUPPORT_UNKNOWN
    pose = torch.tensor([[.05, 0., 0., .1]], dtype=torch.float64)
    a, b = network_inputs([wide], near, pose, 1000)
    c, d = packed_network_inputs([encode_obs1(wide, near)], pose, 1000)
    assert torch.equal(a, c) and torch.equal(b, d)


def test_receiver_repeated_age_evaluation_is_idempotent():
    wide, near = blank_frames(); wide.state[0, 0, 21, 24] = S.OCCUPIED; wide.age_s[0, 0, 21, 24] = .1
    value = decode_obs1(encode_obs1(wide, near))
    a = value.valid_at(1100); b = a.valid_at(1100)
    assert torch.equal(a.wide.age_s, b.wide.age_s)
    assert torch.equal(b.wide.age_s, value.valid_at(1100).wide.age_s)


def test_packet_support_codes_128_to_254_are_valid():
    wide, near = blank_frames(); wide.support[0, 0, 0] = 128; wide.support[0, 0, 1] = 254
    decoded = decode_obs1(encode_obs1(wide, near))
    assert decoded.wide.support[0, 0, :2].tolist() == [128, 254]


def test_packed_no_future_info_or_out_of_order_history():
    wide, near = blank_frames(seq=2)
    old, _ = blank_frames(seq=3, stamp=900)
    with pytest.raises(ValueError, match='newest first'):
        network_inputs([wide, old], near, torch.zeros(1, 4), 1000)
    with pytest.raises(ValueError, match='future'):
        network_inputs([wide], near, torch.zeros(1, 4), 999)


def test_nonzero_packet_padding_rejected_even_with_valid_crc():
    wide, near = blank_frames(); packet = bytearray(encode_obs1(wide, near))
    last_state = HEADER_BYTES+(49*43*3*2+7)//8-1
    packet[last_state] |= 0xC0
    packet[-4:] = struct.pack('<I', zlib.crc32(packet[:-4]))
    with pytest.raises(ValueError, match='padding'): decode_obs1(bytes(packet))


def test_support_floor_and_box_hit_preserve_obstacle_semantics():
    # Floor was previously observed below the present box (not an oracle floor).
    state, _ = ray_evidence(resolve_first_hits(rays_at([.4], dist=.3)), torch.zeros(1, 1, 1), grid=Grid(1, 1, .1, 0., 0.))
    assert state[0, 1, 0, 0] == S.OCCUPIED


def test_warp_history_keeps_sample_support_without_reclassification():
    wide, _ = blank_frames(); wide.state[0, 2, 21, 24] = S.OCCUPIED; wide.age_s[0, 2, 21, 24] = .1
    _, _, support = warp_categorical(wide.state, wide.age_s, wide.sample_pose, wide.sample_pose,
                                      support=wide.support)
    assert torch.equal(support, wide.support)
    assert wide.state[0, 2, 21, 24] == S.OCCUPIED


@pytest.mark.parametrize('segment', ['FLAT', 'PASSAGE', 'TURN_SPOT', None])
def test_route_flat_modes_apply_current_support_tolerance(segment):
    reference, code, valid = grow([[0., .10, .11, .2, .4]], segment_type=segment)
    assert valid.tolist() == [[[True, True, False, False, False]]]
    assert reference[0, 0, -1] == 0


@pytest.mark.parametrize('segment', ['STAIR_UP', 'STAIR_DOWN', 'LANDING'])
def test_route_stair_modes_enable_tread_extension(segment):
    _, _, valid = grow([[0., .2, .4, .6]], segment_type=segment)
    assert bool(valid.all())


def test_mixed_batch_route_is_gdm_input_not_geometry_label():
    height = torch.tensor([[[0., .2, .4]], [[0., .2, .4]]])
    valid = torch.ones_like(height, dtype=torch.bool)
    normals = torch.zeros(*height.shape, 3); normals[..., 2] = 1
    seed = torch.zeros_like(valid); seed[..., 0] = True
    routes = torch.tensor([S.SEGMENT_TYPES.index('FLAT'), S.SEGMENT_TYPES.index('STAIR_UP')])
    h, _, support = grow_support(height, valid, normals, seed, segment_type=routes, current_support_height=torch.zeros(2))
    assert support.tolist() == [[[True, False, False]], [[True, True, True]]]
    assert h[0].tolist() == [[0., 0., 0.]]
    assert torch.allclose(h[1], height[1])


def test_support_normal_threshold_is_local_plane_not_riser_run():
    z = torch.tensor([[[0., .2, .4]]]); valid = torch.ones_like(z, dtype=torch.bool)
    seed = torch.zeros_like(valid); seed[..., 0] = True
    for degrees, expected in [(34., True), (36., False)]:
        angle = math.radians(degrees)
        normal = torch.tensor([math.sin(angle), 0., math.cos(angle)]).expand(1, 1, 3, 3)
        _, _, connected = grow_support(z, valid, normal, seed, segment_type='STAIR_UP')
        assert bool(connected.all()) == expected


def test_support_below_measured_floor_does_not_bridge_platform_gap():
    h, code, valid = grow([[0., 0., -.4, -.4]])
    assert code[0, 0, 2:].tolist() == [255, 255]
    assert not bool(valid[0, 0, 2:].any())


def test_vertical_or_unobserved_hit_cannot_be_tread():
    normal = torch.tensor([[[[0., 0., 1.], [1., 0., 0.], [0., 0., 1.]]]])
    _, _, valid = grow([[0., .2, .4]], normals=normal, segment_type='STAIR_UP')
    assert valid.tolist() == [[[True, False, False]]]


def test_route_hint_invalid_fails_closed():
    with pytest.raises(ValueError, match='segment_type'):
        grow([[0., .2]], segment_type='WRONG')


def test_default_config_and_custom_thresholds(tmp_path):
    from pathlib import Path
    cfg = load_config()
    assert cfg.flat_support_tolerance_m == .1 and cfg.max_z_m == 4.08
    original = Path(__file__).resolve().parents[2]/'configs/gast_nav/maps.yaml'
    modified = tmp_path/'maps.yaml'
    modified.write_text(original.read_text().replace('ttl_s: 0.5', 'ttl_s: 0.3'))
    assert load_config(modified).ttl_s == .3


def test_fine_unknown_or_and_observed_age_are_separate():
    grid, _ = extended_near_grid()
    near = build_near(rays_at([.15], dist=.3), torch.zeros(1, grid.ny, grid.nx))
    assert near.unknown[0, 12, 18] and near.age_s[0, 12, 18] == 0.
    assert torch.isinf(near.age_s[0, 0, 0])


def test_warp_occupancy_preservation_for_rotation_translation_corpus():
    grid = Grid(9, 9, .1, -.4, -.4)
    state = torch.zeros(8, 3, 9, 9, dtype=torch.uint8); state[:, 0, 4, 4] = S.OCCUPIED
    pose = torch.tensor([[.02*math.sin(i), .02*math.cos(i), 0., i*math.pi/7] for i in range(8)])
    out, _, _ = warp_categorical(state, torch.zeros_like(state, dtype=torch.float32), torch.zeros(8, 4), pose, grid=grid)
    assert bool((out == S.OCCUPIED).flatten(1).any(1).all())
    assert not bool((out == S.FREE).any())


def test_analytic_dynamic_geometry_isolated_between_envs():
    r = rays_at([.5], n=3)
    box = DynamicBoxes(torch.tensor([[[.3, 0., .5]], [[.3, 2., .5]], [[.7, 0., .5]]]),
                       torch.full((3, 1, 3), .1), torch.zeros(3, 1))
    resolved = resolve_first_hits(r, boxes=box)
    assert torch.allclose(resolved.distances[:, 0], torch.tensor([.5, load_config().ray_max_range_m, .9]))
    assert resolved.occupied_hit[:, 0].tolist() == [True, False, True]


def test_torch_cpu_batch16_smoke():
    r = full_rays(n=16)
    h = torch.zeros(16, 1, 5)
    state, _ = ray_evidence(resolve_first_hits(r), h, grid=Grid(5, 1, .1, -.2, 0.))
    assert state.device.type == 'cpu' and state.shape == (16, 3, 1, 5)
    assert bool((state == S.FREE).all())


def test_pose4_support_height_tracks_base_z_without_reclassifying_bands():
    mapper = ObstacleMap(1)
    old_pose = torch.tensor([[0., 0., .5, 0.]])
    h = torch.full((1, S.WIDE_NY, S.WIDE_NX), -.5)
    first = mapper.update(rays_at([-.35], dist=.3), h, old_pose, torch.tensor([1000]))
    assert first.state[0, 0, 21, 24] == S.OCCUPIED
    new_pose = torch.tensor([[0., 0., .7, 0.]])
    second = mapper.update(rays_at([-.55]), torch.full_like(h, -.7), new_pose, torch.tensor([1100]))
    assert second.state[0, 0, 21, 24] == S.OCCUPIED
    assert second.support[0, 21, 24] == 15  # world floor unchanged, base is 0.7m above it
    assert second.support_world_z[0, 21, 24] == pytest.approx(0., abs=1e-6)
    assert first.support[0, 21, 24] == 25  # pinned source not rewritten


def test_changed_support_keeps_old_reference_until_occupied_expiry():
    mapper = ObstacleMap(1)
    pose = torch.zeros(1, 4)
    h = torch.zeros(1, S.WIDE_NY, S.WIDE_NX)
    first = mapper.update(rays_at([.15], dist=.3), h, pose, torch.tensor([1000]))
    raised_support = torch.full_like(h, .4)
    rays = full_rays(); origins = rays.origins.clone(); origins[..., 2] += .4
    raised_rays = replace(rays, origins=origins)
    second = mapper.update(raised_rays, raised_support, pose, torch.tensor([1200]))
    assert second.support[0, 21, 24] == 50
    assert second.state[0, 0, 21, 24] == S.OCCUPIED
    assert second.state[0, 1, 21, 24] == S.UNKNOWN  # no free claim on conflicting support
    assert second.age_s[0, 0, 21, 24] == pytest.approx(.2)
    third = mapper.update(raised_rays, raised_support, pose, torch.tensor([1600]))
    assert third.support[0, 21, 24] == 70
    assert third.state[0, 0, 21, 24] == S.FREE
    assert first.support[0, 21, 24] == 50


def test_retained_world_support_avoids_accumulated_2cm_requantization_drift():
    mapper = ObstacleMap(1)
    h = torch.zeros(1, S.WIDE_NY, S.WIDE_NX)
    mapper.update(rays_at([.15], dist=.3), h, torch.zeros(1, 4), torch.tensor([1000]))
    for i in range(1, 6):
        z = .011*i
        pose = torch.tensor([[0., 0., z, 0.]])
        frame = mapper.update(rays_at([.15-z]), torch.full_like(h, -z), pose, torch.tensor([1000+50*i]))
        assert frame.support_world_z[0, 21, 24] == pytest.approx(0., abs=1e-7)
        expected = int(encode_support(torch.tensor([[[-z]]]))[0, 0, 0])
        assert int(frame.support[0, 21, 24]) == expected


def test_obs1_pose4_roundtrip_keeps_base_z_and_yaw():
    wide, near = blank_frames()
    wide = replace(wide, sample_pose=torch.tensor([[1., -2., .43, -.72]], dtype=torch.float64))
    decoded = decode_obs1(encode_obs1(wide, near))
    assert torch.equal(decoded.wide.sample_pose, wide.sample_pose)
    assert len(encode_obs1(wide, near)) == 9426


def test_support_seeds_include_only_valid_body_near_observations():
    from gd_lab.gast_nav.support import make_seed_mask
    contact = torch.tensor([[[0., 0., -.4]]])
    terrain = torch.tensor([[[.1, 0., -.4], [.2, 0., -.4], [.3, 0., -.4]]])
    mask = make_seed_mask(contact, torch.tensor([[True]]), terrain,
                          torch.tensor([[True, False, True]]), torch.tensor([[True, True, False]]))
    assert int(mask.sum()) == 2
    assert mask[0, 21, 24] and mask[0, 21, 25]


def test_self_hit_nearer_than_dynamic_returns_never_carves_through_self():
    r = rays_at([.5], dist=.2, self_hit=True)
    box = DynamicBoxes(torch.tensor([[[.3, 0., .5]]]), torch.full((1, 1, 3), .1), torch.zeros(1, 1))
    resolved = resolve_first_hits(r, boxes=box)
    assert bool(resolved.self_hit.all()) and not bool(resolved.occupied_hit.any())
    assert resolved.distances[0, 0] == pytest.approx(.2)


def test_dynamic_capsule_nearer_than_wall_is_visible():
    r = rays_at([.5], dist=2.)
    cap = DynamicCapsules(torch.tensor([[[.3, 0., 0.]]]), torch.tensor([[[.3, 0., 1.8]]]), torch.tensor([[.2]]))
    resolved = resolve_first_hits(r, capsules=cap)
    assert bool(resolved.occupied_hit.all())
    assert resolved.distances[0, 0] == pytest.approx(.4)


def test_ray_dtype_validation():
    r = rays_at([.2])
    with pytest.raises(ValueError, match='dtypes'):
        resolve_first_hits(replace(r, self_hit=torch.ones(1, 1)))


def test_no_simulator_imported_by_map_modules():
    # Check in a fresh interpreter: other gast_nav tests in this process import the policy side (isaaclab).
    import subprocess, sys
    mods = ('support', 'obstacle_map', 'near_clearance', 'warp', 'packets', 'map_inputs', 'ray_geometry')
    code = ('import sys; ' + '; '.join(f'import gd_lab.gast_nav.{m}' for m in mods) +
            "; print(any(k == 'isaaclab' or k.startswith('omni.') for k in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == 'False'
    assert all(x.device.type == 'cpu' for x in (blank_frames()[0].state, full_rays().origins))
