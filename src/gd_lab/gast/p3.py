"""Opt-in readout fine-tuning; default objective stays unchanged."""
import torch


def configure_train_parts(student, parts):
    if parts not in ('all', 'head', 'head+gate'):
        raise ValueError('unknown train_parts')
    if parts == 'all':
        return list(student.parameters())
    student.requires_grad_(False)
    student.head.requires_grad_(True)
    if parts == 'head+gate':
        student.quality.requires_grad_(True)
    return [p for p in student.parameters() if p.requires_grad]


def restore_optimizer(optimizer, saved, parts):
    previous = saved.get('train_parts', 'all')
    if previous == parts:
        optimizer.load_state_dict(saved['optimizer'])
        return True
    if parts == 'all':
        raise ValueError('partial optimizer cannot resume all')
    # Full optimizer IDs do not map to subsets: explicitly start fresh moments.
    return False


def latent_groups(clean, terrain, seen):
    labels = clean.reshape(-1, 4, 187)
    gap, known = labels[:, 2] > .5, labels[:, 3].bool()
    visible = terrain.reshape(-1, 2, 187)[:, 1] > .5
    ahead = torch.zeros(11, 17, dtype=torch.bool, device=clean.device)
    ahead[2:9, 9:] = True
    ag = gap & known & ahead.flatten()[None]
    has = ag.any(1)
    v = (ag & visible).any(1)
    hidden_seen = seen & ~visible
    never = ~visible & ~seen
    return torch.stack((v, has & ~v & ~(ag & ~hidden_seen).any(1),
                        (ag & never).any(1), known.all(1) & ~gap.any(1)), 1)


def balanced_latent_loss(row_loss, groups):
    """Mean of present group means. Overlapping probe groups stay overlapping.

    Unmatched rows are excluded; all-unmatched batches fall back to plain MSE.
    Existing near-gap floors are not compounded with these weights.
    """
    if groups.shape != (len(row_loss), 4):
        raise ValueError('expected four group masks per row')
    masks = groups.to(row_loss.dtype)
    counts = masks.sum(0)
    present = counts > 0
    weights = (masks / counts.clamp_min(1)).sum(1) / present.sum().clamp_min(1)
    if not present.any():
        return row_loss.mean()
    return (row_loss * weights).sum()


def actor_jacobian_latent_loss(*args, **kwargs):
    # TODO UPDATE10A: select linearization/normalization from diagnostic results.
    raise NotImplementedError('awaiting UPDATE10A actor-Jacobian recommendation')
