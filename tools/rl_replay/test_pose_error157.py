"""CPU regressions for UPDATE157; no Isaac application or GPU."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from pose_error157 import LEVELS, PoseError, from_environ, transform_packet

ROOT = Path(__file__).resolve().parents[2]


class PoseTests(unittest.TestCase):
    def model(self, cfg=LEVELS['P2'], n=3):
        return PoseError(cfg, n, .01, 42, 'cpu')

    def test_off_and_overrides(self):
        self.assertIsNone(from_environ({}))
        self.assertIsNone(from_environ({'U157_WHITE_XY': '1'}))
        self.assertEqual(from_environ({'U157_POSE_ERR': 'P2', 'U157_DELAY_MS': '60'}).delay_ms, 60)
        with self.assertRaises(ValueError):from_environ({'U157_POSE_ERR': 'P4'})
        with self.assertRaises(ValueError):from_environ({'U157_POSE_ERR': 'P2', 'U157_RW_XY': 'nan'})

    def test_p0_hold_and_delay_buffer(self):
        p = self.model(LEVELS['P0'], 1)
        d = self.model(replace(LEVELS['P0'], delay_ms=40), 1)
        for t in range(12):
            gt = torch.tensor([[t*.01, 0., 0.]])
            a = p.update(gt, torch.ones(1))
            b = d.update(gt, torch.ones(1))
            self.assertAlmostEqual(a[0, 0].item(), (t//2)*.02, places=6)
            self.assertAlmostEqual(b[0, 0].item(), max(0, (t//2)*2-4)*.01, places=6)
        p.reset(torch.tensor([0]));d.reset(torch.tensor([0]))
        gt = torch.tensor([[100., 50., 1.]])
        torch.testing.assert_close(d.update(gt, torch.zeros(1)), gt, rtol=0, atol=0)

    def test_movement_gate_white_and_rng(self):
        torch.manual_seed(731)
        before = torch.random.get_rng_state().clone()
        cfg = replace(LEVELS['P2'], k_slip=0., white_xy=0., white_yaw_deg=0., delay_ms=0)
        p = self.model(cfg)
        gt = torch.zeros(3, 3)
        for _ in range(30):
            out = p.update(gt, torch.tensor([0., .05, 0.]), torch.tensor([0., 0., .06]))
        torch.testing.assert_close(out[:2], gt[:2], atol=0, rtol=0)
        self.assertGreater(float(out[2].norm()), 0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        white = self.model(replace(cfg, white_xy=.003))
        self.assertGreater(float(white.update(gt, torch.zeros(3)).norm()), 0)
        self.assertEqual(float(white.drift.norm()), 0.)

    def test_determinism_environment_streams_and_reset(self):
        a, b = self.model(), self.model()
        gt = torch.zeros(3, 3)
        for t in range(31):
            gt[:, 0] = t*.01
            if t == 8:a.reset(torch.tensor([1]))
            aa = a.update(gt, torch.ones(3))
            bb = b.update(gt, torch.ones(3))
            torch.testing.assert_close(aa[[0, 2]], bb[[0, 2]], atol=0, rtol=0)
        # Reset one episode, never let old delayed GT at another location leak.
        a.reset(torch.tensor([0]));gt[0] = torch.tensor([1000., 900., -3.1])
        out = a.update(gt, torch.zeros(3))
        self.assertLess(float((out[0, :2]-gt[0, :2]).norm()), .1)

    def test_summary_relative_windows_do_not_cross_reset(self):
        p = self.model(LEVELS['P0'], 1)
        for t in range(600):
            if t == 300:p.reset(torch.tensor([0]))
            p.update(torch.tensor([[t*.01, 0., 0.]]), torch.ones(1))
        s = p.summary()
        self.assertEqual(len(s['episodes']), 2)
        self.assertIsNone(s['relative_4p8s'])
        self.assertLess(s['overall']['xy_m_max'], .011)

    def test_relative_summary_and_slip_without_yaw_slip(self):
        cfg = replace(LEVELS['P0'], k_slip=.1)
        p = self.model(cfg, 1)
        for t in range(501):
            p.update(torch.tensor([[t*.01, 0., 0.]]), torch.ones(1))
        s = p.summary()
        self.assertIsNotNone(s['relative_4p8s'])
        self.assertGreater(s['overall']['xy_m_max'], 0.)
        self.assertEqual(s['overall']['yaw_deg_max'], 0.)

    def test_packet_transform_preserves_returns_and_roll_pitch(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(42)
        rot = Rotation.from_euler('xyz', [[.1, -.2, .3], [-.2, .1, -.4]])
        xyzw = rot.as_quat();q = xyzw[:, [3, 0, 1, 2]]
        pos = np.array([[10., 20., .6], [30., 40., .5]])
        packet = dict(body_pos=pos, body_quat=q, origins=pos[:, None]+rng.normal(0, .1, (2, 4, 3)),
                      dirs=rng.normal(size=(2, 4, 2, 3, 3)), ranges=np.ones((2, 4, 2, 3)),
                      valid=np.ones((2, 4, 2, 3), bool))
        odom = np.array([[11., 19., .5], [31., 39., -.7]])
        out = transform_packet(packet, odom)
        np.testing.assert_array_equal(out['body_pos'][:, :2], odom[:, :2])
        np.testing.assert_array_equal(out['body_pos'][:, 2], pos[:, 2])
        np.testing.assert_array_equal(out['origins'][..., 2], packet['origins'][..., 2])
        self.assertIs(out['ranges'], packet['ranges']);self.assertIs(out['valid'], packet['valid'])
        np.testing.assert_allclose(np.linalg.norm(out['dirs'], axis=-1), np.linalg.norm(packet['dirs'], axis=-1))
        new = Rotation.from_quat(out['body_quat'][:, [1, 2, 3, 0]]).as_euler('xyz')
        np.testing.assert_allclose(new[:, :2], rot.as_euler('xyz')[:, :2], atol=1e-14)
        np.testing.assert_allclose(new[:, 2], odom[:, 2], atol=1e-14)
        np.testing.assert_array_equal(packet['body_pos'], pos)

    def test_history_off_bit_exact_against_prechange_method(self):
        def load(relative, names, namespace, legacy=False):
            tree = ast.parse((ROOT/relative).read_text())
            nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
            if legacy:
                for node in nodes:
                    if not isinstance(node, ast.ClassDef):continue
                    for fn in node.body:
                        if isinstance(fn, ast.FunctionDef) and fn.name == 'update':
                            fn.args.args = fn.args.args[:-1];fn.args.defaults = []
                            for line in fn.body:
                                if isinstance(line, ast.Assign) and isinstance(line.value, ast.IfExp):
                                    if isinstance(line.targets[0], ast.Name) and line.targets[0].id == 'pose':line.value = line.value.body
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), relative, 'exec'), namespace)
        for rayir in (False, True):
            spaces = []
            for legacy in (False, True):
                ns = dict(torch=torch, F=F, HISTORY_OFFSETS=(48,32,16,8,4,2,1,0),
                          pose_xyyaw=lambda e:e.pose, CELLS=187, CHANNELS=3)
                load('src/gd_lab/gast/geometry.py', ['warp_memory'], ns)
                load('src/gd_lab/gast/observations.py', ['TerrainHistory'], ns, legacy)
                if rayir:load('tools/rl_replay/rayir_e1.py', ['RayIrHistory'], ns, legacy)
                spaces.append(ns)
            env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.01,
                                  common_step_counter=0, pose=torch.zeros(2, 3))
            key = 'RayIrHistory' if rayir else 'TerrainHistory'
            a, b = [s[key](env) for s in spaces]
            for t in range(55):
                env.common_step_counter = t;env.pose = torch.tensor([[t*.01, 0., t*.001], [0., t*.01, -t*.002]])
                width = 561 if rayir else 374
                terrain = torch.ones(2, width);terrain[:, :187] = t*.01
                if t == 29:a.reset(torch.tensor([0]));b.reset(torch.tensor([0]))
                self.assertTrue(torch.equal(a.update(env, terrain), b.update(env, terrain)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
