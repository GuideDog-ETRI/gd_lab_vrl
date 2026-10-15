"""CPU test of behavior_events on a synthetic recording with known events and reward terms."""
import math
import unittest

import numpy as np

import behavior_events as be


def recording():
    T, gx = 300, np.linspace(-0.8, 0.8, 17)
    feet = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    env = {k: [] for k in ("bodies", "scan", "foot_contact", "done", "base_quat", "terms", "reward", "advantage")}
    for t in range(T):
        bx = 0.01 * t
        xs, ys = np.meshgrid(bx + gx, np.linspace(-0.5, 0.5, 11))
        zs = np.where((xs > 1.6) & (xs < 1.8), -0.65, 0.0)          # a pit at x 1.6..1.8
        env["scan"].append(np.stack((xs, ys, zs), -1).reshape(-1, 3).tolist())
        z = np.full(4, 0.02)
        contact = [1, 1, 1, 1]
        fx = np.array([bx + .35, bx + .35, bx - .35, bx - .35])
        if 140 <= t < 160:                                            # front-right foot sinks into the pit
            fx[0], z[0] = 1.7, -0.20
        if 95 <= t < 105:                                             # both hind feet in the air
            contact[2] = contact[3] = 0; z[2] = z[3] = 0.10
        if 200 <= t < 206:                                            # front-left foot kicks high
            z[1], contact[1] = 0.35, 0
        env["bodies"].append([[0, 0, .5]] + [[fx[i], (-.15, .15, -.15, .15)[i], z[i]] for i in range(4)])
        env["foot_contact"].append(contact)
        env["done"].append(1 if t == 290 else 0)
        env["base_quat"].append([1, 0, 0, 0])
        terms = [0.01, 0.0, 0.0]
        if 135 <= t < 165: terms[1] = -0.05                            # "drop_penalty" around the drop
        if 90 <= t < 110: terms[2] = 0.04                              # "hop_bonus" around the hop
        env["terms"].append(terms); env["reward"].append(sum(terms)); env["advantage"].append(0.0)
    meta = {"term_names": ["track", "drop_penalty", "hop_bonus"], "foot_names": feet,
            "body_names": ["trunk"] + feet, "gamma": .99, "lam": .95, "num_envs": 1, "steps": T, "dt": .01}
    return {"meta": meta, "envs": [env]}


class BehaviorEvents(unittest.TestCase):
    def test_detect_and_attribute(self):
        report = be.analyze(recording())
        ev = report["events"]
        self.assertEqual(ev["foot_drop"]["count"], 1)
        self.assertEqual(ev["foot_drop"]["occurrences"][0]["start"], 140)
        self.assertEqual(ev["hind_hop"]["count"], 1)
        self.assertEqual(ev["high_lift"]["count"], 1)
        self.assertEqual(ev["fall"]["count"], 0)              # done at level stance (t=290) = time-out, not a fall
        self.assertEqual(ev["foot_drop"]["terms"][0]["term"], "drop_penalty")   # most negative first
        self.assertEqual(ev["hind_hop"]["terms"][-1]["term"], "hop_bonus")      # most positive last
        self.assertIn("발 빠짐", be.markdown(report, "synthetic"))

    def test_share_sums_to_reward_return(self):
        rec = recording(); env, meta = rec["envs"][0], rec["meta"]
        share = be.advantage_share(env, meta)
        k = meta["gamma"] * meta["lam"]; acc = 0.0
        for t in range(len(env["reward"]) - 1, -1, -1):
            acc = env["reward"][t] + k * (1 - env["done"][t]) * acc
        self.assertAlmostEqual(share[0].sum(), acc, places=9)


if __name__ == "__main__":
    unittest.main()


class ActionEventTests(unittest.TestCase):
    """UPDATE138 review: liftoff/touchdown never link two episodes, both feet at once count once, chatter ignored."""

    def setUp(self):
        self.rec = recording()
        self.env, self.meta = self.rec["envs"][0], self.rec["meta"]

    def test_hind_liftoff_once_for_both_feet_and_touchdown_matches(self):
        ev = be.detect(self.env, self.meta)
        self.assertEqual(ev["liftoff_hind"], [(94, 95)])      # before-contact row 94 -> row 95 airborne (no 'after' field)
        self.assertEqual(ev["touchdown_hind"], [(104, 105)])

    def test_after_contact_is_used_and_reset_does_not_create_events(self):
        env = dict(self.env)
        contact = np.array(env["foot_contact"])
        contact[291:, 0] = 0                                   # new episode after done at 290 starts with FR in the air
        env["foot_contact"] = contact.tolist()
        after = contact.copy(); after[:-1] = contact[1:]; after[290] = contact[290]   # pre-reset 'after' row
        env["foot_contact_after"] = after.tolist()
        ev = be.detect(env, self.meta)
        self.assertFalse(any(s == 290 for s, _ in ev["liftoff_front"]))

    def test_chatter_is_ignored(self):
        env = dict(self.env)
        contact = np.array(env["foot_contact"]); contact[50, 0] = 0   # one-row flicker
        env["foot_contact"] = contact.tolist()
        ev = be.detect(env, self.meta)
        self.assertNotIn((49, 50), ev["liftoff_front"])

    def test_unknown_foot_names_are_refused(self):
        meta = dict(self.meta, foot_names=["LF_foot", "RF_foot", "LH_foot", "RH_foot"])
        with self.assertRaises(ValueError):
            be.detect(self.env, meta)

    def test_dominance_is_per_episode_and_mean_free(self):
        share = be.advantage_share(self.env, self.meta)
        dev, dom = be.dominance(share, self.env["done"])
        self.assertLess(abs(dev[:291].mean(0)).max(), 1e-12)
        self.assertLess(abs(dev[291:].mean(0)).max(), 1e-12)
        self.assertEqual(dom[150], 1)                          # inside the drop_penalty window


class FallTests(unittest.TestCase):
    def test_tilted_done_is_a_fall(self):
        rec = recording(); env = rec["envs"][0]
        q = math.radians(40) / 2
        env["base_quat"] = [list(x) for x in env["base_quat"]]
        env["base_quat"][290] = [math.cos(q), math.sin(q), 0, 0]   # 40 deg roll at the done
        self.assertEqual(len(be.detect(env, rec["meta"])["fall"]), 1)
