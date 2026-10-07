"""BIVT-Ray -> GAST teacher warm start and per-group PPO learning rates (CPU, no simulator)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from tensordict import TensorDict

from gd_lab.gast import warm_start as ws

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "checkpoints/teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005"


def _gast_policy():
    from torch import nn

    from gd_lab.gast.temporal import TemporalTerrainEncoder
    from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC, POLICY_OBS_DIM
    from gd_lab.teachers.cvtt.actor_critic import DreamwaqVrlActorCritic

    # GastActorCritic without its Isaac imports (gd_lab.gast.teacher needs a running Isaac app);
    # test_gast_actor_critic_adds_only_the_encoder_and_decoder pins that the real class adds exactly these.
    class GastActorCritic(DreamwaqVrlActorCritic):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.terrain_encoder = TemporalTerrainEncoder()
            self.terrain_decoder = nn.Sequential(nn.Linear(32, 128), nn.ELU(), nn.Linear(128, 187*6))

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    cfg = dict(yaml.load((PACKAGE / "params/agent.yaml").read_text(), Loader=Loader)["policy"])
    cfg.pop("class_name")
    cfg.setdefault("height_scan_start", DREAMWAQ_SPEC.critic.offset("height_scan"))
    critic_dim = DREAMWAQ_SPEC.critic.resolve(height_scan=187).total
    obs = TensorDict({"policy": torch.zeros(1, POLICY_OBS_DIM), "critic": torch.zeros(1, critic_dim)}, [1])
    return GastActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **cfg)


@pytest.fixture(scope="module")
def checkpoint():
    path = PACKAGE / "teacher/21068_top1.pt"
    if not path.is_file():
        pytest.skip("21068 teacher package not present")
    return path


def test_every_backbone_tensor_comes_from_bivt_and_only_the_terrain_encoder_is_fresh(checkpoint):
    policy = _gast_policy()
    bivt = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    merged, report = ws.bivt_to_gast_state_dict(bivt, policy.state_dict())
    assert report["copied"] > 40 and report["fresh_terrain"] > 0
    for name, value in merged.items():
        if ws.is_terrain(name):
            continue
        assert torch.equal(value, bivt[name].to(value.dtype)), name
    assert all(name.startswith("terrain_encoder.") for name in bivt if name not in merged or ws.is_terrain(name))


def test_mismatched_backbone_is_refused():
    gast = {"actor.0.weight": torch.zeros(4, 3), "terrain_encoder.head.0.weight": torch.zeros(2, 2)}
    with pytest.raises(ValueError, match="lacks"):
        ws.bivt_to_gast_state_dict({}, gast)
    with pytest.raises(ValueError, match="!="):
        ws.bivt_to_gast_state_dict({"actor.0.weight": torch.zeros(4, 4)}, gast)
    with pytest.raises(ValueError, match="no GAST counterpart"):
        ws.bivt_to_gast_state_dict({"actor.0.weight": torch.zeros(4, 3), "extra.w": torch.zeros(1)}, gast)


def test_warm_start_groups_rates_zeroes_the_latent_and_restores_cenet_state(checkpoint):
    policy = _gast_policy()
    template = torch.optim.Adam(policy.parameters(), lr=1e-3)
    alg = SimpleNamespace(policy=policy, optimizer=template, schedule="fixed", learning_rate=1e-3)
    manifest = ws.warm_start_from_bivt(alg, str(checkpoint), 1e-4, 1e-3)
    assert manifest["source_iteration"] == 21068 and manifest["source_camera_profile"] == "vendor_new"
    assert manifest["learning_rates"] == {"policy": 1e-4, "terrain": 1e-3}
    assert manifest["cenet_optimizer_restored"] and manifest["latent_head_zeroed"]
    groups = {g[ws.GROUP_KEY]: {id(p) for p in g["params"]} for g in alg.optimizer.param_groups}
    terrain = {id(p) for n, p in policy.named_parameters() if ws.is_terrain(n)}
    assert groups["terrain"] == terrain
    assert not groups["policy"] & terrain
    assert not any(id(p) in groups["policy"] | groups["terrain"] for p in policy.cenet.parameters())
    history = torch.randn(3, 8 * 375)
    assert torch.count_nonzero(policy.terrain_encoder(history)) == 0  # latent starts at exactly 0
    with pytest.raises(ValueError, match="fixed"):
        ws.warm_start_from_bivt(SimpleNamespace(policy=policy, optimizer=template, schedule="adaptive"),
                                str(checkpoint), 1e-4, 1e-3)


def test_group_rates_survive_a_state_round_trip_and_a_gast_rollback():
    policy = _gast_policy()
    optimizer = ws.build_grouped_optimizer(policy, 1e-4, 1e-3, torch.optim.Adam(policy.parameters(), lr=5e-4))
    fresh = ws.build_grouped_optimizer(policy, 9.0, 9.0, torch.optim.Adam(policy.parameters(), lr=5e-4))
    fresh.load_state_dict(optimizer.state_dict())
    assert ws.group_learning_rates(fresh) == {"policy": 1e-4, "terrain": 1e-3}
    source = (ROOT / "src/gd_lab/gast/teacher.py").read_text()
    assert "if not is_grouped(self.optimizer)" in source  # GastPPO._restore keeps per-group rates
    runner = (ROOT / "src/gd_lab/rl/runner.py").read_text()
    assert 'if not any("gd_lab_group" in group' in runner  # runner.load keeps per-group rates


def test_gap_clean_task_matches_the_bivt_clean_weights():
    source = (ROOT / "src/gd_lab/gast/tasks.py").read_text()
    assert "task='GastGapClean'" in source
    block = source.split("class GastGapCleanTeacherCfg", 1)[1].split("registry.register_task", 1)[0]
    assert "intrusion_weight: float = -3.0" in block and "clean_weight: float = 1.5" in block
    for term in ("GapMonitor", "gap_intrusion_penalty", "gap_clean_bonus", "platform_gap_diagnostics",
                 "GapMetadataTerrainGenerator"):
        assert term in block


def test_gast_actor_critic_adds_only_the_encoder_and_decoder():
    source = (ROOT / "src/gd_lab/gast/teacher.py").read_text()
    init = source.split("class GastActorCritic(DreamwaqVrlActorCritic):", 1)[1].split("def terrain_latent", 1)[0]
    init = init.split("def __init__(self, *args, **kwargs):", 1)[1]
    assert "self.terrain_encoder = TemporalTerrainEncoder()" in init
    assert "self.terrain_decoder = nn.Sequential(nn.Linear(32, 128), nn.ELU(), nn.Linear(128, 187*6))" in init
    assert init.count("self.") == 2


def test_zero_latent_head_still_trains_the_encoder_after_one_step():
    """Codex review 10/06: with a zero head the upstream encoder gets no gradient on the very first
    step; the head itself does, and from the second step on the whole encoder learns."""
    torch.manual_seed(0)
    policy = _gast_policy()
    ws.zero_latent_head(policy)
    optimizer = ws.build_grouped_optimizer(policy, 1e-4, 1e-3, torch.optim.Adam(policy.parameters(), lr=1e-3))
    encoder, history = policy.terrain_encoder, torch.randn(32, 8 * 375)
    columns = torch.randn(policy.actor[0].weight.shape[0], 32) * 0.1  # trained latent columns are non-zero
    upstream = [p for n, p in encoder.named_parameters() if not n.startswith("head.")]
    flows = []
    for _ in range(2):
        optimizer.zero_grad()
        latent = encoder(history)
        (((latent @ columns.T) ** 2).mean() + latent.sum()).backward()
        flows.append((float(encoder.head[0].weight.grad.abs().sum()),
                      sum(float(p.grad.abs().sum()) for p in upstream if p.grad is not None)))
        optimizer.step()
    assert flows[0][0] > 0 and flows[0][1] == 0
    assert flows[1][0] > 0 and flows[1][1] > 0


# --- Codex static review 10/06: source contract, resume records, launcher paths -----------------------------

def _checkpoint_dict(checkpoint):
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def test_source_teacher_contract_is_checked_not_assumed(checkpoint):
    import copy
    import hashlib

    loaded = _checkpoint_dict(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ws.validate_source(loaded, digest)
    ws.validate_source(loaded, digest, expected_sha256=digest.upper())
    with pytest.raises(ValueError, match="sha256"):
        ws.validate_source(loaded, digest, expected_sha256="0" * 64)
    mutations = {
        "task": lambda g: g["gap_finetune"].__setitem__("task", "Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0"),
        "camera": lambda g: g["gap_finetune"].__setitem__("camera_profile", "vendor_legacy"),
        "observation": lambda g: g["observation_context"].__setitem__("version", "bivt_ray_occlusion_v1"),
        "missing gap record": lambda g: g.pop("gap_finetune"),
        "missing observation record": lambda g: g.pop("observation_context"),
    }
    for name, mutate in mutations.items():
        broken = {"infos": {"gd_lab": copy.deepcopy(loaded["infos"]["gd_lab"])}}
        mutate(broken["infos"]["gd_lab"])
        with pytest.raises(ValueError, match="contract"):
            ws.validate_source(broken, digest)


def test_resume_checks_the_restored_rates_and_keeps_the_original_provenance(tmp_path):
    policy = _gast_policy()
    saved = ws.build_grouped_optimizer(policy, 1e-4, 1e-3, torch.optim.Adam(policy.parameters(), lr=5e-4))
    restored = ws.build_grouped_optimizer(policy, 1e-4, 1e-3, torch.optim.Adam(policy.parameters(), lr=5e-4))
    restored.load_state_dict(saved.state_dict())
    assert ws.check_resumed_rates(restored, 1e-4, 1e-3) == {"policy": 1e-4, "terrain": 1e-3}
    with pytest.raises(ValueError, match="differ"):
        ws.check_resumed_rates(restored, 2e-4, 1e-3)  # CLI rate != restored rate
    origin = {"source": "a.pt", "source_sha256": "f" * 64, "learning_rates": {"policy": 1e-4, "terrain": 1e-3}}
    first = tmp_path / "model_100.pt"
    torch.save({"iter": 100, "infos": {"gd_lab": {"gast_warm_start": origin}}}, first)
    record = ws.resumed_provenance(str(first))
    assert record["origin"] == origin and record["resumed_from"]["iteration"] == 100
    second = tmp_path / "model_200.pt"
    torch.save({"iter": 200, "infos": {"gd_lab": {"gast_warm_start": {**record, "task": "x"}}}}, second)
    again = ws.resumed_provenance(str(second))
    assert again["origin"] == origin and again["resumed_from"]["iteration"] == 200  # the original survives resumes
    bare = tmp_path / "model_300.pt"
    torch.save({"iter": 300, "infos": {"gd_lab": {}}}, bare)
    with pytest.raises(ValueError, match="no gast_warm_start"):
        ws.resumed_provenance(str(bare))
    trainer = (ROOT / "scripts/gast/train_teacher.py").read_text()
    assert trainer.index("runner.load(resume_path)") < trainer.index("check_resumed_rates(")


def _run_launcher(tmp_path, *args, sha=None):
    import os
    import subprocess

    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "apptainer").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{tmp_path}/apptainer.args"\n')
    (stub / "apptainer").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "GAST_BIVT_TEACHER_SHA256"}
    env.update(PATH=f"{stub}:{env['PATH']}", GAST_RUN_ID=f"pytest_launcher_{tmp_path.name}")
    if sha:
        env["GAST_BIVT_TEACHER_SHA256"] = sha
    try:
        return subprocess.run(["bash", "experiments/gast/train_teacher_gapclean_from_bivt_3gpu.sh", *args], cwd=ROOT,
                              env=env, capture_output=True, text=True, timeout=120)
    finally:
        import shutil
        shutil.rmtree(ROOT / "logs/launches" / env["GAST_RUN_ID"], ignore_errors=True)


def test_launcher_works_from_the_repo_root_with_relative_paths(tmp_path, checkpoint):
    relative = str(checkpoint.relative_to(ROOT))  # caller-relative, resolved before the launcher changes directory
    result = _run_launcher(tmp_path, "smoke", relative)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "apptainer.args").read_text().split("\n")
    assert args[args.index("--warm_start_bivt") + 1] == str(checkpoint)
    assert "--warm_start_sha256" not in args
    result = _run_launcher(tmp_path, "train", relative)
    assert result.returncode != 0 and "GAST_BIVT_TEACHER_SHA256" in result.stderr
    result = _run_launcher(tmp_path, "train")
    assert result.returncode != 0 and "teacher path" in result.stderr
    result = _run_launcher(tmp_path, "train", relative, sha="ab" * 32)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "apptainer.args").read_text().split("\n")
    assert args[args.index("--warm_start_sha256") + 1] == "ab" * 32
    assert "--total_envs" in args and args[args.index("--total_envs") + 1] == "4096"
    for launcher in ("train_teacher_gapclean_from_bivt_3gpu.sh", "train_teacher_3gpu.sh"):
        text = (ROOT / "experiments/gast" / launcher).read_text()
        assert 'cp "$0"' not in text and text.index('realpath "${BASH_SOURCE[0]}"') < text.index('cd "$root"')
