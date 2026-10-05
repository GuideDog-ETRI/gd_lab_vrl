from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from gd_lab.rl.cenet import CENet
from gd_lab.rl.distributed_sync import (
    assert_synchronized_state,
    raise_if_any,
    synchronize_empirical_normalizer,
)
from gd_lab.rl.ppo import DreamwaqPPO
from gd_lab.rl.runner import DreamwaqRunner


class ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(3, 2)
        self.critic = nn.Linear(3, 1)
        self.terrain_encoder = nn.Linear(2, 2)
        self.cenet = CENet(
            input_dim=4,
            obs_dim_per_step=4,
            latent_dim=2,
            velocity_dim=2,
            encoder_hidden_dims=(8,),
            decoder_hidden_dims=(8,),
            adaboot_enabled=True,
            adaboot_min_updates=0,
        )


class ToyNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, 1))
        self.register_buffer("_var", torch.ones(1, 1))
        self.register_buffer("_std", torch.ones(1, 1))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.until = None

    def update(self, samples: torch.Tensor) -> None:
        raise AssertionError("the distributed path must reduce global moments")


def _worker(rank: int, port: int, checkpoint_path: str) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(1234)
        policy = ToyPolicy()
        ppo_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        algorithm = SimpleNamespace(policy=policy, rnd=None, device="cpu")

        # Exercise the same hook called by upstream RSL-RL PPO.update().
        for parameter in (*policy.actor.parameters(), *policy.critic.parameters(), *policy.terrain_encoder.parameters()):
            parameter.grad = torch.full_like(parameter, 1.0 + 2.0 * rank)
        DreamwaqPPO.reduce_parameters(algorithm)
        for parameter in (*policy.actor.parameters(), *policy.critic.parameters(), *policy.terrain_encoder.parameters()):
            assert torch.all(parameter.grad == 2.0)
        assert all(parameter.grad is None for parameter in policy.cenet.parameters())

        from gd_lab.rl.distributed_sync import install_collective_optimizer_step
        install_collective_optimizer_step(ppo_optimizer, device="cpu", context="test PPO optimizer")
        ppo_optimizer.step()

        # Both ranks merge different observation batches into exactly the same
        # EmpiricalNormalization moments.
        norm = ToyNormalizer()
        samples = torch.zeros(2, 1) if rank == 0 else torch.full((2, 1), 2.0)
        synchronize_empirical_normalizer(norm, samples)
        assert norm.count.item() == 4
        assert torch.allclose(norm._mean, torch.tensor([[1.0]]))
        assert torch.allclose(norm._var, torch.tensor([[1.0]]))

        policy.cenet.adaboot_update_count.fill_(1)
        policy.cenet.resample_adaboot()
        coin = [torch.empty_like(policy.cenet.inject_gt) for _ in range(2)]
        dist.all_gather(coin, policy.cenet.inject_gt)
        assert torch.equal(coin[0], coin[1])

        opts = [("ppo", ppo_optimizer), ("cenet", policy.cenet.optimizer)]
        before = assert_synchronized_state(policy, opts, context="pre-CENet test")

        # A NaN on just one rank must make every rank skip the CENet optimizer.
        history = torch.randn(8, 4)
        if rank == 1:
            history[0, 0] = float("nan")
        metrics = policy.cenet.update(
            history,
            torch.randn(8, 2),
            torch.randn(8, 4),
            torch.ones(8, dtype=torch.bool),
        )
        assert metrics["cenet_skipped_steps"] == 1.0
        assert assert_synchronized_state(policy, opts, context="collective CENet skip") == before

        # A subsequent finite CENet update averages its gradients and initializes
        # identical Adam slots on every rank.
        policy.cenet.update(torch.randn(8, 4), torch.randn(8, 2), torch.randn(8, 4))
        assert_synchronized_state(policy, opts, context="CENet optimizer update")

        if rank == 0:
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": ppo_optimizer.state_dict(),
                    "cenet_optimizer_state_dict": policy.cenet.optimizer.state_dict(),
                },
                checkpoint_path,
            )
        dist.barrier()

        # Perturb each rank differently, then use the real DreamwaqRunner.load
        # collective wrapper around checkpoint restoration.
        with torch.no_grad():
            policy.actor.weight.add_(rank + 1.0)
        for group in ppo_optimizer.param_groups:
            group["lr"] += rank * 1e-3

        def load_local(path, load_optimizer=True, map_location=None):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            policy.load_state_dict(saved["model_state_dict"])
            if load_optimizer:
                ppo_optimizer.load_state_dict(saved["optimizer_state_dict"])
                policy.cenet.optimizer.load_state_dict(saved["cenet_optimizer_state_dict"])
            return {"resume_test": True}

        runner = SimpleNamespace(
            device="cpu",
            alg=SimpleNamespace(policy=policy, optimizer=ppo_optimizer),
            _load_local=load_local,
        )
        infos = DreamwaqRunner.load(runner, checkpoint_path)
        assert infos == {"resume_test": True}
        assert_synchronized_state(policy, opts, context="checkpoint resume test")

        # One rank's failure is raised with the same message on both ranks.
        messages = []
        try:
            raise_if_any("injected failure" if rank == 1 else None, device="cpu", context="test rank error")
        except RuntimeError as exc:
            messages = [str(exc), None]
        gathered_messages = [None, None]
        dist.all_gather_object(gathered_messages, messages[0])
        assert gathered_messages[0] == gathered_messages[1]
        assert "rank 1: injected failure" in gathered_messages[0]
    finally:
        dist.destroy_process_group()


def test_bivt_distributed_ppo_cenet_and_resume_are_collectively_synchronized(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    checkpoint_path = str(tmp_path / "distributed_resume.pt")
    mp.start_processes(
        _worker,
        args=(port, checkpoint_path),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
