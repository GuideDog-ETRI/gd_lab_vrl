"""Frozen blind anchor and bounded visual residual. No simulator dependency."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.distributions import Normal

from gd_lab.rl.actor_critic import DreamwaqActorCritic
from gd_lab.students.gavd.model import GridAttentionStudent
from gd_lab.core.camera_contract import DEFAULT_CAMERA_PROFILE


@dataclass(frozen=True)
class ResidualConfig:
    # Units are policy actions, NOT radians; physical correction = action_scale * delta.
    bound: float = .2
    slew: float = .02
    stale_seconds: float = .25
    initial_std: float = .1
    camera_profile: str = DEFAULT_CAMERA_PROFILE

    def __post_init__(self):
        if not all(torch.isfinite(torch.tensor(x)) and x > 0 for x in
                   (self.bound, self.slew, self.stale_seconds, self.initial_std)):
            raise ValueError("Residual limits and initial_std must be finite and positive")


class _ConfigLoader(yaml.SafeLoader):
    pass


_ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                              lambda loader, node: tuple(loader.construct_sequence(node)))


def load_blind_teacher(checkpoint, agent_yaml, obs):
    """Strictly reconstruct the supplied DWB; never initialize a vision teacher."""
    config = yaml.load(Path(agent_yaml).read_text(), Loader=_ConfigLoader)
    policy = dict(config["policy"])
    if policy.pop("class_name") != "gd_lab.rl.actor_critic:DreamwaqActorCritic":
        raise ValueError("BAVRL requires a blind DreamwaqActorCritic checkpoint")
    device = obs[config["obs_groups"]["policy"][0]].device
    teacher = DreamwaqActorCritic(obs, config["obs_groups"], 12, **policy).to(device)
    checkpoint = Path(checkpoint)
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    teacher.load_state_dict(data["model_state_dict"], strict=True)
    teacher.requires_grad_(False).eval()
    return teacher, sha256(checkpoint.read_bytes()).hexdigest()


class BAVRL(nn.Module):
    def __init__(self, teacher, config=None):
        super().__init__()
        config = config or ResidualConfig()
        self.config = config
        self.teacher = teacher.requires_grad_(False).eval()
        self.vision = GridAttentionStudent(camera_profile=config.camera_profile)
        width = teacher.actor[0].in_features
        self.residual = nn.Sequential(nn.Linear(width + 32 + 12 + 1, 128), nn.ELU(),
                                      nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 12))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.log_std = nn.Parameter(torch.full((12,), float(torch.log(torch.tensor(config.initial_std)))))
        # Value network is an independent copy, not the frozen anchor critic.
        self.critic = deepcopy(teacher.critic).requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # Includes both observation normalizers and CENet.
        return self

    def forward(self, obs, frames, hidden, previous, age, available, vision_age=None):
        with torch.no_grad():
            features = self.teacher._actor_input(obs, inference=True)
            base = self.teacher.actor(features)
            critic_obs = self.teacher.critic_obs_normalizer(self.teacher.get_critic_obs(obs))
        good = available.bool() & torch.isfinite(age) & (age >= 0) & (age <= self.config.stale_seconds)
        if frames is None:
            good = torch.zeros_like(good)
        else:
            good = good & torch.isfinite(frames).flatten(1).all(1)
        memory = hidden.clone()
        safe_age = torch.nan_to_num(age, nan=self.config.stale_seconds,
                                    posinf=self.config.stale_seconds, neginf=self.config.stale_seconds)
        memory[:, 63] = ((safe_age if vision_age is None else vision_age) / self.config.stale_seconds).clamp(0, 1)
        # Missing cameras bypass the visual network entirely, including its memory.
        latent = base.new_zeros((len(base), 32))
        next_hidden = base.new_zeros((len(base), 64))
        spatial = base.new_zeros((len(base), 187, 5))
        if good.any():
            latent[good], next_hidden[good], spatial[good] = self.vision.encode(frames[good], memory[good])
        mean = self.residual(torch.cat((features, latent, previous,
                                       (safe_age / self.config.stale_seconds).clamp(0, 1)[:, None]), -1))
        distribution = Normal(mean, self.log_std.clamp(-6, 0).exp().expand_as(mean))
        return base, distribution, self.critic(critic_obs).squeeze(-1), next_hidden, spatial, good

    def compose(self, base, raw, previous, good):
        target = self.config.bound * raw.tanh()
        delta = previous + (target - previous).clamp(-self.config.slew, self.config.slew)
        delta = delta.clamp(-self.config.bound, self.config.bound)
        # Invalid input immediately removes residual; this safety fallback overrides slew.
        delta = torch.where(good[:, None], delta, torch.zeros_like(delta))
        return base + delta, delta

    def checkpoint(self, teacher_hash, optimizer=None):
        return {"schema": "bavrl_v1", "teacher_sha256": teacher_hash,
                "config": asdict(self.config), "model": self.state_dict(),
                "optimizer": None if optimizer is None else optimizer.state_dict()}

    def restore(self, data, teacher_hash, optimizer=None):
        if (data.get("schema") != "bavrl_v1" or data.get("teacher_sha256") != teacher_hash
                or data.get("config") != asdict(self.config)):
            raise ValueError("BAVRL schema, teacher hash or residual configuration mismatch")
        # Refuse a checkpoint that tries to replace the anchor under the same hash.
        for key, value in self.teacher.state_dict().items():
            if not torch.equal(value, data["model"]["teacher." + key].to(value.device)):
                raise ValueError("Frozen teacher state mismatch: " + key)
        self.load_state_dict(data["model"], strict=True)
        if optimizer is not None and data["optimizer"] is not None:
            optimizer.load_state_dict(data["optimizer"])
        self.teacher.eval()
