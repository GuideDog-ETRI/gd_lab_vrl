"""Raycast visibility geometry with an injected cast function (CPU, no simulator)."""

from __future__ import annotations

import torch
from gd_lab_blind_start.raycast_visibility import raycast_visible_points

K = torch.tensor([[40.0, 0.0, 40.0], [0.0, 40.0, 22.5], [0.0, 0.0, 1.0]])
IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # camera frame == world, optical axis +z


def _points():
    return torch.tensor([[[0.0, 0.0, 1.0],     # centre, clear
                          [0.1, 0.0, 1.0],     # occluded by the fake wall
                          [0.0, 0.0, -1.0],    # behind the camera
                          [5.0, 0.0, 1.0],     # outside the frame
                          [0.0, 0.0, 9.0]]])   # beyond the depth clip


def test_frame_clip_and_occlusion():
    def cast(starts, directions, max_dist):
        # Ground plane at optical depth 1 m; a wall at 0.5 m covers rays with x > 0.05.
        depth = torch.where(directions[:, 0] > 0.05, 0.5, 1.0)
        return depth / directions[:, 2]

    visible = raycast_visible_points(_points(), torch.zeros(1, 3), IDENTITY, K, (45, 80), (0.15, 5.0), cast)
    assert visible.tolist() == [[True, False, False, False, False]]


def test_neighbour_pixels_on_the_same_plane_count_as_visible():
    def cast(starts, directions, max_dist):
        return 1.0 / directions[:, 2]  # every pixel sees the plane at optical depth 1 m

    points = torch.tensor([[[0.0, 0.0, 1.0]]])
    assert raycast_visible_points(points, torch.zeros(1, 3), IDENTITY, K, (45, 80), (0.15, 5.0), cast).all()


def test_leg_capsule_blocks_only_rays_passing_through_it():
    def cast(starts, directions, max_dist):
        return 1.0 / directions[:, 2]

    points = torch.tensor([[[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]]])
    # Vertical capsule at x=0, z=0.5 (between camera and the first point only).
    seg_a = torch.tensor([[[0.0, -0.2, 0.5]]])
    seg_b = torch.tensor([[[0.0, 0.2, 0.5]]])
    occluders = (seg_a, seg_b, torch.tensor([0.05]))
    visible = raycast_visible_points(points, torch.zeros(1, 3), IDENTITY, K, (45, 80), (0.15, 5.0), cast,
                                     occluders=occluders)
    assert visible.tolist() == [[False, True]]
