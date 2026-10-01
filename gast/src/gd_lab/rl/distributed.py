"""Collectives for auxiliary optimizers and observation normalization."""
import hashlib
import torch
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized()


def any_rank(value, device):
    flag = torch.tensor(int(bool(value)), device=device)
    if active():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def average_gradients(parameters):
    if not active():
        return
    params = list(parameters)
    present = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    used = [p for p, flag in zip(params, present.tolist()) if flag]
    if not used:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in used])
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    offset = 0
    for p in used:
        p.grad = flat[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()


@torch.no_grad()
def update_normalizer(normalizer, values):
    if not active():
        normalizer.update(values)
        return
    if not normalizer.training or (normalizer.until is not None and normalizer.count >= normalizer.until):
        return
    x = values.double()
    moments = torch.cat((x.sum(0).flatten(), x.square().sum(0).flatten(), x.new_tensor([x.shape[0]])))
    dist.all_reduce(moments)
    width = x.shape[-1]
    n = moments[-1]
    mean = (moments[:width]/n).reshape_as(normalizer._mean)
    var = (moments[width:2*width]/n-mean.flatten().square()).clamp_min(0).reshape_as(normalizer._var)
    old_mean = normalizer._mean.double()
    old_var = normalizer._var.double()
    count = normalizer.count.double()+n
    rate = n/count
    delta = mean-old_mean
    new_mean = old_mean+rate*delta
    new_var = old_var+rate*(var-old_var+delta*(mean-new_mean))
    normalizer._mean.copy_(new_mean)
    normalizer._var.copy_(new_var)
    normalizer._std.copy_(new_var.sqrt())
    normalizer.count.copy_(count)


def assert_synchronized(policy, optimizers):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            t = item.detach().cpu().contiguous()
            if t.is_floating_point() and not torch.isfinite(t).all():
                raise RuntimeError('Non-finite training state')
            digest.update(t.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for value in item: visit(value)
        else:
            digest.update(repr(item).encode())
    visit(policy.state_dict())
    for optimizer in optimizers: visit(optimizer.state_dict())
    fingerprint = digest.hexdigest()
    if active():
        hashes = [None]*dist.get_world_size()
        dist.all_gather_object(hashes, fingerprint)
        if len(set(hashes)) != 1:
            raise RuntimeError(f'Distributed training states diverged: {hashes}')
    return fingerprint
