#!/usr/bin/env python3
"""E2 model conversion (CPU, torch only).

  convert_models.py --to v2 SRC OUT     any GAST Ray+IR lineage checkpoint -> GAST-Narrow v2 shape (E2 trials)
  convert_models.py --to e1 SRC OUT     -> 3906/6566 shape for the E1 harness (LiDAR map zero, margin m folded)

Shapes (actor first layer in, critic first layer in): Ray+IR 6566/3906 (235, 298), Narrow v1 (251, 304),
Narrow v2 (252, 307). Actor input = [base 235, LiDAR latent 16, margin 1]; critic = [base 298, privileged 6 or 9].
--to v2 adds zero columns (and a zero LiDAR encoder for 6566), so the converted policy acts exactly as the source.
--to e1 folds the constant latent c = enc(zero map) and the margin obs into the first-layer bias (exact up to
float rounding): actor.0(x) = W[:, :235] x + b + W[:, 235:251] c + W[:, 251] * margin_obs.
"""
import argparse
import json

import torch
from torch import nn

BASE_ACTOR, BASE_CRITIC, LIDAR, FRAMES, NY, NX = 235, 298, 16, 2, 30, 48
ENCODER = {"cnn.0.weight": (8, 2, 5, 5), "cnn.0.bias": (8,), "cnn.2.weight": (16, 8, 3, 3), "cnn.2.bias": (16,),
           "cnn.4.weight": (32, 16, 3, 3), "cnn.4.bias": (32,), "head.0.weight": (64, 768), "head.0.bias": (64,),
           "head.2.weight": (16, 64), "head.2.bias": (16,)}


def kind_of(sd):
    a, c = sd["actor.0.weight"].shape[1], sd["critic.0.weight"].shape[1]
    kinds = {(235, 298): "rayir", (251, 304): "narrow_v1", (252, 307): "narrow_v2"}
    if (a, c) not in kinds:
        raise ValueError(f"unknown checkpoint shape actor.0 in={a} critic.0 in={c}")
    return kinds[(a, c)]


def widen(w, total):
    out = torch.zeros(w.shape[0], total, dtype=w.dtype)
    out[:, :w.shape[1]] = w
    return out


def to_v2(sd):
    kind = kind_of(sd)
    new = {k: v.clone() for k, v in sd.items()}
    if kind == "rayir":
        for name, shape in ENCODER.items():
            new["lidar_encoder." + name] = torch.zeros(shape)
    new["actor.0.weight"] = widen(sd["actor.0.weight"], BASE_ACTOR + LIDAR + 1)
    new["critic.0.weight"] = widen(sd["critic.0.weight"], BASE_CRITIC + 9)
    return kind, new


def encoder_zero_latent(sd):
    cnn = nn.Sequential(nn.Conv2d(FRAMES, 8, 5, stride=2, padding=2), nn.ELU(), nn.Conv2d(8, 16, 3, stride=2, padding=1),
                        nn.ELU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(), nn.Flatten())
    head = nn.Sequential(nn.Linear(768, 64), nn.ELU(), nn.Linear(64, LIDAR))
    cnn.load_state_dict({k[len("lidar_encoder.cnn."):]: v for k, v in sd.items() if k.startswith("lidar_encoder.cnn.")})
    head.load_state_dict({k[len("lidar_encoder.head."):]: v for k, v in sd.items() if k.startswith("lidar_encoder.head.")})
    with torch.no_grad():
        return head(cnn(torch.zeros(1, FRAMES, NY, NX)))[0]


def to_e1(sd, margin_obs=0.0):
    kind = kind_of(sd)
    if kind == "rayir":
        return kind, {k: v.clone() for k, v in sd.items()}, 0.0
    c = encoder_zero_latent(sd)
    W, b = sd["actor.0.weight"], sd["actor.0.bias"]
    tail = torch.cat((c, torch.full((W.shape[1] - BASE_ACTOR - LIDAR,), float(margin_obs))))
    new = {k: v.clone() for k, v in sd.items() if not k.startswith("lidar_encoder.")}
    with torch.no_grad():
        new["actor.0.weight"] = W[:, :BASE_ACTOR].clone()
        new["actor.0.bias"] = b + W[:, BASE_ACTOR:] @ tail
        new["critic.0.weight"] = sd["critic.0.weight"][:, :BASE_CRITIC].clone()
        x = torch.randn(64, BASE_ACTOR)
        full = torch.cat((x, tail.expand(64, -1)), 1) @ W.T + b
        err = (full - (x @ new["actor.0.weight"].T + new["actor.0.bias"])).abs().max().item()
    if err > 1e-4:
        raise RuntimeError(f"fold error {err}")
    return kind, new, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--to", choices=("v2", "e1"), required=True)
    p.add_argument("--margin_obs", type=float, default=0.0, help="e1: folded margin obs (m - 0.10) / 0.05")
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    if a.to == "v2":
        kind, new = to_v2(sd)
        info = {"kind": kind}
    else:
        kind, new, err = to_e1(sd, a.margin_obs)
        info = {"kind": kind, "fold_err": err, "lidar": "zero", "margin_obs": a.margin_obs}
    torch.save({"model_state_dict": new, "iter": ck.get("iter"), "infos": {"e2_converted_from": a.src, **info}}, a.out)
    print(json.dumps({"src": a.src, "out": a.out, "to": a.to, **info}))


if __name__ == "__main__":
    main()
