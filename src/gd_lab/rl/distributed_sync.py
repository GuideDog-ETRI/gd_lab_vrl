"""Collective-safe helpers for the project-specific distributed PPO state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import torch
import torch.distributed as dist


def distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def any_rank(flag: bool, device: torch.device | str) -> bool:
    """Return a rank-identical OR; every participant must call this in the same order."""
    if not distributed_active():
        return bool(flag)
    value = torch.tensor(int(bool(flag)), dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Average stable gradients, treating rank-local missing gradients as zero."""
    if not distributed_active():
        return
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        return
    world_size = dist.get_world_size()
    presence = torch.tensor(
        [int(parameter.grad is not None) for parameter in parameters],
        dtype=torch.int32,
        device=parameters[0].device,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    active = [parameter for parameter, present in zip(parameters, presence.tolist(), strict=True) if present]
    if not active:
        return
    for parameter in active:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for parameter in active:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
        offset += count


def synchronize_empirical_normalizer(normalizer: torch.nn.Module, samples: torch.Tensor) -> None:
    """Update running moments from the union of every rank's sample batch."""
    if not normalizer.training:
        return
    if normalizer.until is not None and normalizer.count >= normalizer.until:
        return
    if not distributed_active():
        normalizer.update(samples)
        return
    batch_count = torch.tensor(samples.shape[0], dtype=torch.long, device=samples.device)
    batch_sum = samples.sum(dim=0, keepdim=True)
    batch_sum_sq = samples.square().sum(dim=0, keepdim=True)
    dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_sum_sq, op=dist.ReduceOp.SUM)
    if not batch_count.item():
        return
    batch_mean = batch_sum / batch_count
    batch_var = (batch_sum_sq / batch_count - batch_mean.square()).clamp_min_(0)
    old_count = normalizer.count.clone()
    total_count = old_count + batch_count
    delta = batch_mean - normalizer._mean
    mean = normalizer._mean + delta * (batch_count / total_count)
    var = (
        normalizer._var * old_count
        + batch_var * batch_count
        + delta.square() * (old_count * batch_count / total_count)
    ) / total_count
    normalizer._mean.copy_(mean)
    normalizer._var.copy_(var)
    normalizer._std.copy_(var.sqrt())
    normalizer.count.copy_(total_count)


def install_collective_optimizer_step(
    optimizer: torch.optim.Optimizer, *, device: torch.device | str, context: str
) -> None:
    """Make a local optimizer exception become the same exception on every rank."""
    if getattr(optimizer, "_gd_lab_collective_step", False):
        return
    original_step = optimizer.step

    def collective_step(*args, **kwargs):
        error = None
        result = None
        try:
            result = original_step(*args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        raise_if_any(error, device=device, context=context)
        return result

    optimizer.step = collective_step
    optimizer._gd_lab_collective_step = True
def raise_if_any(error: str | None, *, device: torch.device | str, context: str) -> None:
    """Make local optimizer failures fail identically after a shared decision."""
    if not any_rank(error is not None, device):
        return
    if distributed_active():
        messages: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(messages, error)
    else:
        messages = [error]
    details = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(messages) if message)
    raise RuntimeError(f"{context} failed on at least one distributed rank ({details})")


def _feed_hash(digest, value) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _feed_hash(digest, key)
            _feed_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _feed_hash(digest, item)
    else:
        digest.update(type(value).__qualname__.encode())
        digest.update(b"\0")
        digest.update(json.dumps(value, sort_keys=True, default=repr, allow_nan=True).encode())


def state_fingerprint(policy: torch.nn.Module, optimizers: Iterable[tuple[str, torch.optim.Optimizer]]) -> str:
    """Hash persistent policy state and optimizer groups/slots by parameter name."""
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        _feed_hash(digest, name)
        _feed_hash(digest, value)
    parameter_names = {id(parameter): name for name, parameter in policy.named_parameters()}
    for optimizer_name, optimizer in optimizers:
        _feed_hash(digest, optimizer_name)
        for group in optimizer.param_groups:
            names = [parameter_names.get(id(parameter), f"<unknown:{tuple(parameter.shape)}>") for parameter in group["params"]]
            options = {key: value for key, value in group.items() if key != "params"}
            _feed_hash(digest, (names, options))
        states = []
        for parameter, state in optimizer.state.items():
            states.append((parameter_names.get(id(parameter), f"<unknown:{tuple(parameter.shape)}>"), state))
        for name, state in sorted(states, key=lambda item: item[0]):
            _feed_hash(digest, name)
            _feed_hash(digest, state)
    return digest.hexdigest()


def assert_synchronized_state(
    policy: torch.nn.Module,
    optimizers: Iterable[tuple[str, torch.optim.Optimizer]],
    *,
    context: str,
) -> str:
    """Verify every rank has identical policy and optimizer state; return its hash."""
    fingerprint = state_fingerprint(policy, optimizers)
    if not distributed_active():
        return fingerprint
    fingerprints: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(fingerprints, fingerprint)
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        details = ", ".join(f"rank {rank}={value}" for rank, value in enumerate(fingerprints))
        raise RuntimeError(f"Distributed state mismatch after {context}: {details}")
    return fingerprint
