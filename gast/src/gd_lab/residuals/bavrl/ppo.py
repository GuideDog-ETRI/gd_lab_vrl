"""PPO over raw residual samples; deterministic clipping is part of the actuator.

Memory inputs are detached at camera boundaries (one-capture truncated gradients).
Do not call this full sequence BPTT. Replay must use exactly the collected frames,
memory, age, previous residual and availability; never re-augment minibatches.
"""

import torch

from gd_lab.students.gavd.model import spatial_loss


def gae(rewards, values, next_values, dones, gamma=.99, lam=.95):
    """next_values must be zero for true terminals, final-state V for timeouts."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] - values[step]
        carry = delta + gamma * lam * (~dones[step]).float() * carry
        result[step] = carry
    return result, result + values


class ResidualPPO:
    def __init__(self, model, lr=3e-4, clip=.2, spatial_coef=.1):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.clip = clip
        self.spatial_coef = spatial_coef

    def update(self, batch):
        base, dist, value, _, spatial, good = self.model(
            batch["obs"], batch["frames"], batch["hidden"], batch["previous"],
            batch["age"], batch["available"], batch.get("vision_age"))
        log_prob = dist.log_prob(batch["raw"]).sum(-1)
        ratio = (log_prob - batch["log_prob"]).exp()
        advantage = batch["advantage"]
        if good.any():
            a = advantage[good]
            advantage = (advantage - a.mean()) / a.std(unbiased=False).clamp_min(1e-8)
            surrogate = torch.minimum(ratio * advantage,
                                      ratio.clamp(1-self.clip, 1+self.clip) * advantage)
            actor_loss = -surrogate[good].mean()
            entropy = dist.entropy().sum(-1)[good].mean()
            _, delta = self.model.compose(base, dist.mean, batch["previous"], good)
            penalty = delta[good].square().mean() + (delta[good]-batch["previous"][good]).square().mean()
        else:
            actor_loss = log_prob.sum() * 0
            entropy = penalty = actor_loss
        value_loss = (value-batch["return"]).square().mean()
        supervised = good & batch["supervised"].bool()
        auxiliary = spatial_loss(spatial[supervised], batch["terrain"][supervised]) if supervised.any() else spatial.sum()*0
        loss = actor_loss + .5*value_loss - .001*entropy + .01*penalty + self.spatial_coef*auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BAVRL loss; refusing optimizer step")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
        self.optimizer.step()
        return {"loss": loss.item(), "actor": actor_loss.item(), "value": value_loss.item(),
                "spatial": auxiliary.item(), "grad_norm": grad.item(),
                "ratio_mean": ratio.detach().mean().item(), "valid_fraction": good.float().mean().item()}
