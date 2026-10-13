import os
import torch
from torch import nn

from .geometry import reconstruction_loss
from .temporal import TemporalTerrainEncoder


def rank():
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else 0


def tensor_report(x):
    x = x.detach()
    finite = torch.isfinite(x)
    good = x[finite]
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "bad": int((~finite).sum()),
        "min": float(good.min()) if good.numel() else None,
        "max": float(good.max()) if good.numel() else None,
    }


@torch.no_grad()
def grad_report(named_params, label, verbose=False):
    finite = True
    norms32, norms64 = [], []
    for name, p in named_params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        ok = bool(torch.isfinite(g).all())
        finite = finite and ok
        norms32.append(torch.linalg.vector_norm(g.float()))
        norms64.append(torch.linalg.vector_norm(g.double()))
        if verbose or not ok:
            bad = (~torch.isfinite(g)).nonzero()[:8].cpu().tolist()
            print(f"[NAN_GRAD] rank={rank()} stage={label} param={name} "
                  f"stats={tensor_report(g)} bad_indices={bad}", flush=True)
    device = norms64[0].device if norms64 else torch.device("cpu")
    n32 = (torch.linalg.vector_norm(torch.stack(norms32))
           if norms32 else torch.zeros((), device=device))
    n64 = (torch.linalg.vector_norm(torch.stack(norms64))
           if norms64 else torch.zeros((), device=device, dtype=torch.float64))
    result = {"finite": finite, "norm32": float(n32), "norm64": float(n64)}
    if verbose or not finite or not bool(torch.isfinite(n32)):
        print(f"[NAN_NORM] rank={rank()} stage={label} {result}", flush=True)
    return result


@torch.no_grad()
def clip_from_norm64(params, norm64, max_norm=1.0):
    # Caller must first establish finite gradients and finite norm64.
    scale = min(1.0, max_norm / (norm64 + 1e-6))
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def install_hooks(module):
    handles = []

    def backward_hook(name):
        def hook(mod, grad_input, grad_output):
            ins = list(_tensors(grad_input))
            outs = list(_tensors(grad_output))
            input_bad = any(not bool(torch.isfinite(t).all()) for t in ins)
            output_bad = any(not bool(torch.isfinite(t).all()) for t in outs)
            if input_bad or output_bad:
                print(f"[NAN_BWD] rank={rank()} module={name} "
                      f"grad_output_bad={output_bad} grad_input_bad={input_bad}",
                      flush=True)
        return hook

    def forward_hook(name):
        def hook(mod, args, output):
            for i, t in enumerate(_tensors(output)):
                if not bool(torch.isfinite(t).all()):
                    print(f"[NAN_FWD] rank={rank()} module={name} "
                          f"output={i} stats={tensor_report(t)}", flush=True)
            if isinstance(mod, nn.LayerNorm):
                x = args[0].detach().double()
                var = x.var(-1, unbiased=False)
                print(f"[NAN_LN] rank={rank()} module={name} "
                      f"input={tensor_report(args[0])} "
                      f"var_min={float(var.min())} eps={mod.eps}", flush=True)
        return hook

    for name, child in module.named_modules():
        # Leaf modules plus GRU/MHA. Container hooks add little localization.
        if not list(child.children()) or isinstance(
            child, (nn.GRU, nn.MultiheadAttention)
        ):
            handles.append(child.register_forward_hook(forward_hook(name)))
            handles.append(child.register_full_backward_hook(backward_hook(name)))
    return handles


def replay_aux(policy, history, clean, reference_pred):
    # No optimizer, no collectives, no mutation of the live policy.
    # Preserve the complete failing local batch in CPU RAM.
    payload = {
        "history": history.detach().cpu().clone(),
        "clean": clean.detach().cpu().clone(),
        "pred": reference_pred.detach().cpu().clone(),
        "encoder": {
            k: v.detach().cpu().clone()
            for k, v in policy.terrain_encoder.state_dict().items()
        },
        "decoder": {
            k: v.detach().cpu().clone()
            for k, v in policy.terrain_decoder.state_dict().items()
        },
    }
    # Accessible for interactive inspection; never torch.save().
    policy._nan_aux_payload = payload
    seq = payload["history"].reshape(-1, 8, 375)
    print(f"[NAN_ENV] rank={rank()} torch={torch.__version__} "
          f"cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
          f"gpu={torch.cuda.get_device_name(history.device) if history.is_cuda else 'cpu'} "
          f"history={tensor_report(seq)} clean={tensor_report(payload['clean'])}",
          flush=True)
    for name, value in (
        ("height", seq[..., :187]),
        ("visibility", seq[..., 187:374]),
        ("age", seq[..., -1]),
    ):
        print(f"[NAN_INPUT] rank={rank()} channel={name} "
              f"{tensor_report(value)}", flush=True)
    print(f"[NAN_INPUT] rank={rank()} "
          f"all_zero_rows={int((seq == 0).flatten(1).all(1).sum())}", flush=True)

    # Exact dtype/autocast assumptions must be explicit.
    # The supplied teacher code has no autocast context.
    if history.dtype != torch.float32:
        raise RuntimeError("Extend replay for the actual training dtype first")
    if torch.is_autocast_enabled():
        raise RuntimeError("Capture/replay the actual autocast configuration first")

    cases = [
        ("cuda_native", history.device, torch.float32,
         policy.terrain_encoder.math_sdpa,
         policy.terrain_encoder.disable_cudnn_gru),
        ("cuda_math_only", history.device, torch.float32, True, False),
        ("cuda_gru_only", history.device, torch.float32, False, True),
        ("cuda_math_and_gru", history.device, torch.float32, True, True),
        ("cpu_fp32", torch.device("cpu"), torch.float32, True, True),
        ("cpu_fp64", torch.device("cpu"), torch.float64, True, True),
    ]
    for label, device, dtype, math_only, no_cudnn in cases:
        handles = []
        enc = dec = model = x = target = pred = loss = None
        try:
            enc = TemporalTerrainEncoder().to(device=device, dtype=dtype)
            enc.execution_chunk_size = policy.terrain_encoder.execution_chunk_size
            enc.math_sdpa = math_only
            enc.disable_cudnn_gru = no_cudnn
            enc.load_state_dict(payload["encoder"])
            enc.train(policy.terrain_encoder.training)
            dec = nn.Sequential(nn.Linear(32, 128), nn.ELU(),
                                nn.Linear(128, 187 * 6)).to(
                                    device=device, dtype=dtype)
            dec.load_state_dict(payload["decoder"])
            dec.train(policy.terrain_decoder.training)
            model = nn.ModuleDict({"encoder": enc, "decoder": dec})
            handles = install_hooks(model)
            x = payload["history"].to(device=device, dtype=dtype).detach().clone()
            # Enables full input-gradient localization in replay.
            x.requires_grad_(True)
            target = payload["clean"].to(device=device, dtype=dtype)
            pred = dec(enc(x)).reshape(-1, 187, 6)
            loss = reconstruction_loss(pred, target)
            (0.1 * loss).backward()
            stats = grad_report(model.named_parameters(), label, verbose=True)
            delta = (pred.detach().cpu().double()
                     - payload["pred"].double()).abs().max()
            print(f"[NAN_REPLAY] rank={rank()} case={label} "
                  f"loss={float(loss.detach())} pred_delta={float(delta)} "
                  f"input_grad={tensor_report(x.grad)} grads={stats}", flush=True)
        except Exception as exc:
            print(f"[NAN_REPLAY_ERROR] rank={rank()} case={label} "
                  f"type={type(exc).__name__} error={exc}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
            del enc, dec, model, x, target, pred, loss
