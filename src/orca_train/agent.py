from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .replay import ReplayBuffer


def schedule(spec: str | float, step: int) -> float:
    if isinstance(spec, (float, int)):
        value = float(spec)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Schedule value must be finite and non-negative")
        return value
    match = re.fullmatch(r"linear\(([^,]+),([^,]+),([^,]+)\)", spec)
    if not match:
        try:
            value = float(spec)
        except ValueError as error:
            raise ValueError(f"Invalid schedule: {spec!r}") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Schedule value must be finite and non-negative")
        return value
    start, end, duration = map(float, match.groups())
    if not all(math.isfinite(value) for value in (start, end, duration)):
        raise ValueError("Linear schedule values must be finite")
    if start < 0.0 or end < 0.0 or duration <= 0.0:
        raise ValueError(
            "Linear schedule endpoints must be non-negative and duration positive"
        )
    mix = np.clip(step / duration, 0.0, 1.0)
    return float((1.0 - mix) * start + mix * end)


def sample_noisy_action(
    mean: torch.Tensor,
    stddev: float,
    clip: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    noise = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    noise = (noise * stddev).clamp(-clip, clip)
    return (mean + noise).clamp(-1.0, 1.0)


def actor_loss_from_q_values(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return -torch.minimum(q1, q2).mean()


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (self.pad,) * 4, mode="replicate")
        limit = 2 * self.pad + 1
        offsets = torch.randint(limit, (images.shape[0], 2), device=images.device)
        return torch.stack(
            [
                padded[i, :, y : y + images.shape[-2], x : x + images.shape[-1]]
                for i, (y, x) in enumerate(offsets.tolist())
            ]
        )


class Encoder(nn.Module):
    def __init__(self, pixel_shape: tuple[int, int, int], feature_dim: int):
        super().__init__()
        channels = pixel_shape[0]
        self.convs = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = int(np.prod(self.convs(torch.zeros(1, *pixel_shape)).shape[1:]))
        self.projection = nn.Sequential(nn.Linear(flat_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float().div(255.0).sub(0.5)
        return self.projection(self.convs(pixels).flatten(1))


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.policy = mlp(observation_dim, hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy(observation))


class Critic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.q1 = mlp(observation_dim + action_dim, hidden_dim, 1)
        self.q2 = mlp(observation_dim + action_dim, hidden_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((observation, action), dim=-1)
        return self.q1(inputs), self.q2(inputs)


@dataclass(frozen=True)
class DrQV2Config:
    feature_dim: int = 50
    hidden_dim: int = 1024
    learning_rate: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_schedule: str = "linear(1.0,0.1,500000)"
    stddev_clip: float = 0.3
    batch_size: int = 256
    update_every_steps: int = 2
    augmentation_pad: int = 4


class DrQV2Agent:
    def __init__(
        self,
        pixel_shape: tuple[int, int, int],
        proprio_dim: int,
        action_dim: int,
        device: str | torch.device,
        config: DrQV2Config | None = None,
    ):
        self.device = torch.device(device)
        self.config = config or DrQV2Config()
        self.action_dim = action_dim
        self.encoder = Encoder(pixel_shape, self.config.feature_dim).to(self.device)
        observation_dim = self.config.feature_dim + proprio_dim
        self.actor = Actor(observation_dim, self.config.hidden_dim, action_dim).to(self.device)
        self.critic = Critic(observation_dim, self.config.hidden_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.augmentation = RandomShiftsAug(self.config.augmentation_pad)

    def _features(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.encoder(pixels), proprio.float()), dim=-1)

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray], step: int, eval_mode: bool = False) -> np.ndarray:
        pixels = torch.as_tensor(observation["pixels"], device=self.device).unsqueeze(0)
        proprio = torch.as_tensor(observation["proprio"], device=self.device).unsqueeze(0)
        action = self.actor(self._features(pixels, proprio))
        if not eval_mode:
            stddev = schedule(self.config.stddev_schedule, step)
            action = action + torch.randn_like(action) * stddev
        return action.clamp(-1.0, 1.0).squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer, step: int) -> dict[str, float]:
        if step % self.config.update_every_steps:
            return {}
        batch = replay.sample(self.config.batch_size, self.device)
        pixels = self.augmentation(batch.pixels.float())
        next_pixels = self.augmentation(batch.next_pixels.float())
        observation = self._features(pixels, batch.proprio)

        with torch.no_grad():
            next_observation = self._features(next_pixels, batch.next_proprio)
            next_action = self.actor(next_observation)
            stddev = schedule(self.config.stddev_schedule, step)
            next_action = sample_noisy_action(
                next_action, stddev, self.config.stddev_clip
            )
            target_q1, target_q2 = self.critic_target(next_observation, next_action)
            target_q = batch.rewards + batch.discounts * torch.minimum(target_q1, target_q2)

        q1, q2 = self.critic(observation, batch.actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.encoder_optimizer.step()
        self.critic_optimizer.step()

        detached_observation = observation.detach()
        action = sample_noisy_action(
            self.actor(detached_observation), stddev, self.config.stddev_clip
        )
        actor_q1, actor_q2 = self.critic(detached_observation, action)
        actor_loss = actor_loss_from_q_values(actor_q1, actor_q2)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for parameter, target_parameter in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_parameter.lerp_(parameter, self.config.critic_target_tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q": float(q1.mean().item()),
            "target_q": float(target_q.mean().item()),
        }

    def save(self, path: str | Path, step: int) -> None:
        torch.save(
            {
                "step": step,
                "config": asdict(self.config),
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "encoder_optimizer": self.encoder_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> int:
        state = torch.load(path, map_location=self.device, weights_only=True)
        for name in ("encoder", "actor", "critic", "critic_target"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("encoder_optimizer", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        return int(state["step"])
