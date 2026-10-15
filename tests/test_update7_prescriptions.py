"""One CPU test per UPDATE7 distillation prescription."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspect sys.modules while decorating.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


weights = load_file(
    "update7_test_weights", "src/gd_lab/gast/update7_weights.py"
)
launcher = load_file(
    "update7_test_launcher", "scripts/gast/launch_update7.py"
)


def labels():
    clean = torch.zeros(6, 4, 11, 17)
    clean[:, 3] = 1
    terrain = torch.zeros(6, 2, 11, 17)
    # Row 0: known hidden ahead corridor gap.
    clean[0, 2, 3, 10] = 1
    # Row 1: known visible ahead corridor gap.
    clean[1, 2, 3, 10] = 1
    terrain[1, 1, 3, 10] = 1
    # Row 2: hidden gap behind.
    clean[2, 2, 3, 8] = 1
    # Row 3: hidden ahead gap outside the corridor.
    clean[3, 2, 0, 10] = 1
    # Row 4: unknown gap; must never select a row.
    clean[4, 2, 3, 10] = 1
    clean[4, 3, 3, 10] = 0
    # Row 5: flat.
    near = (
        (clean[:, 2] > .5) & clean[:, 3].bool()
    ).flatten(1).any(1)
    return clean.flatten(1), terrain.flatten(1), near


class PrescriptionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_a_ghost_zero_is_identity_without_geometry_or_rng(self):
        camera_geometry = types.ModuleType("gd_lab.core.camera_geometry")
        camera_geometry.depth_pixels_world = mock.Mock(
            side_effect=AssertionError("disabled ghost must not project")
        )
        with mock.patch.dict(
            sys.modules,
            {"gd_lab.core.camera_geometry": camera_geometry},
        ):
            ghost_module = load_file(
                "update7_test_ghost", "src/gd_lab/mdp/platform_gap_noise.py"
            )
        self.assertEqual(ghost_module.PlatformGapDepthGhostCfg().probability, .25)
        ghost = ghost_module.PlatformGapDepthGhost(
            ghost_module.PlatformGapDepthGhostCfg(probability=0.)
        )
        frames = torch.rand(2, 4, 2, 5, 7)
        snapshot = (
            frames,
            torch.ones(2, 4, 5, 7),
            torch.zeros(2, 4, 3),
            torch.zeros(2, 4, 3, 3),
            torch.zeros(2, 4, 3, 3),
        )
        # Fresh state, as in a ghost-off training run (an in-flight ghost still finishes: tests/test_vrl_camera_fixes.py).
        rng = torch.get_rng_state().clone()
        result = ghost(
            frames, snapshot, torch.zeros(2, 3),
            torch.ones(2, dtype=torch.bool),
        )
        self.assertIs(result, frames)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIsNone(ghost.remaining)
        self.assertEqual(
            launcher.prescription_flags("a", 3., 1.),
            ["--gap_ghost_probability", "0", "--gap_observed_supervision"],
        )
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                ghost_module.PlatformGapDepthGhostCfg(probability=value)

    def test_b_hidden_ahead_latent_floor_and_gradient(self):
        clean, terrain, near = labels()
        base = 1 + (2. - 1) * near.float()
        latent, action, hidden = weights.objective_row_weights(
            base, near, clean, terrain, hidden_gap_latent_weight=4.
        )
        self.assertEqual(
            hidden.tolist(), [True, False, False, False, False, False]
        )
        self.assertEqual(latent.tolist(), [4., 2., 2., 2., 1., 1.])
        self.assertTrue(torch.equal(action, base))
        # Existing gap_weight=2 and requested floor=4 give 4, never 8.
        losses = torch.arange(6, dtype=torch.float32, requires_grad=True)
        loss = (losses * latent).sum() / latent.sum()
        loss.backward()
        self.assertTrue(torch.allclose(losses.grad, latent / latent.sum()))
        default_latent, default_action, _ = weights.objective_row_weights(
            base, near, clean, terrain
        )
        self.assertTrue(torch.equal(default_latent, base))
        self.assertTrue(torch.equal(default_action, base))
        self.assertEqual(
            launcher.prescription_flags("b", 4., 2.),
            ["--hidden_gap_latent_weight", "4.0"],
        )
        with self.assertRaises(ValueError):
            launcher.prescription_flags("b", 2., 2.)

    def test_c_near_gap_action_floor_and_present_row_reduction(self):
        clean, terrain, near = labels()
        base = 1 + (2. - 1) * near.float()
        latent, action, _ = weights.objective_row_weights(
            base, near, clean, terrain, gap_action_weight=4.
        )
        self.assertTrue(torch.equal(latent, base))
        self.assertEqual(action.tolist(), [4., 4., 4., 4., 1., 1.])
        present = torch.tensor([True, False, True, True, True, True])
        actual = torch.zeros(6, 2, requires_grad=True)
        expected = torch.ones(6, 2)
        row_loss = (actual[present] - expected[present]).square().mean(-1)
        loss = (
            row_loss * action[present]
        ).sum() / action[present].sum()
        self.assertAlmostEqual(float(loss), 1.)
        loss.backward()
        self.assertTrue(torch.equal(actual.grad[1], torch.zeros(2)))
        expected_gradient = (
            -action[present] / action[present].sum()
        )[:, None].expand(-1, 2)
        self.assertTrue(torch.allclose(actual.grad[present], expected_gradient))
        self.assertEqual(
            launcher.prescription_flags("c", 4., 2.),
            ["--gap_action_weight", "4.0"],
        )
        for invalid in (0., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                weights.row_weight_floor(base, near, invalid)


    def test_ahead_gap_rows_select_visible_and_hidden_ahead_gaps_only(self):
        import torch
        from gd_lab.gast import update7_weights as w
        clean = torch.zeros(3, 4, 11, 17)
        clean[:, 3] = 1
        clean[0, 2, 5, 12] = 1          # ahead corridor gap
        clean[1, 2, 5, 3] = 1           # gap behind the body: not ahead
        terrain = torch.zeros(3, 2, 11, 17)
        terrain[0, 1, 5, 12] = 1        # visible: still selected (unlike hidden_ahead_gap_rows)
        rows = w.ahead_gap_rows(clean.flatten(1))
        assert rows.tolist() == [True, False, False]
        base = torch.tensor([4., 4., 1.])
        lat, act, hidden = w.objective_row_weights(base, torch.tensor([True, True, False]), clean.flatten(1),
                                                   terrain.flatten(1), ahead_gap_latent_weight=16.)
        assert lat.tolist() == [16., 4., 1.] and act.tolist() == [4., 4., 1.] and hidden.tolist() == [False, False, False]

    def test_launcher_refuses_mixing_the_ahead_floor(self):
        for extra in (["--ahead_gap_latent_weight", "16"], ["--ahead_gap_latent_weight=16"]):
            with self.assertRaises((SystemExit, ValueError)):
                launcher.build_command("b", 16., "python", extra)


if __name__ == "__main__":
    unittest.main()
