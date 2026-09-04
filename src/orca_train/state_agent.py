from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .agent import (
    Actor,
    Critic,
    actor_loss_from_q_values,
    sample_noisy_action,
    schedule,
)
from .replay import StateReplayBuffer


@dataclass(frozen=True)
class StateAgentConfig:
    hidden_dim: int = 1024
    learning_rate: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_schedule: str = "linear(0.2,0.05,100000)"
    stddev_clip: float = 0.2
    batch_size: int = 256
    update_every_steps: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.stddev_clip) or self.stddev_clip < 0.0:
            raise ValueError("stddev_clip must be finite and non-negative")
        try:
            schedule(self.stddev_schedule, step=0)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid stddev_schedule: {self.stddev_schedule!r}"
            ) from error


class StateAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str | torch.device,
        config: StateAgentConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.config = config or StateAgentConfig()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.actor = Actor(
            self.state_dim, self.config.hidden_dim, self.action_dim
        ).to(self.device)
        self.critic = Critic(
            self.state_dim, self.config.hidden_dim, self.action_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, np.ndarray],
        step: int,
        eval_mode: bool = False,
    ) -> np.ndarray:
        state_array = np.asarray(observation["state"], dtype=np.float32)
        if state_array.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.state_dim},), got {state_array.shape}"
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("State observation must contain only finite values")
        state = torch.as_tensor(state_array, device=self.device).unsqueeze(0)
        action = self.actor(state)
        if not eval_mode:
            stddev = schedule(self.config.stddev_schedule, step)
            action = sample_noisy_action(
                action, stddev, self.config.stddev_clip
            )
        return action.clamp(-1.0, 1.0).squeeze(0).cpu().numpy()

    def update(self, replay: StateReplayBuffer, step: int) -> dict[str, float]:
        if step % self.config.update_every_steps:
            return {}
        batch = replay.sample(self.config.batch_size, self.device)

        with torch.no_grad():
            next_action = self.actor(batch.next_states)
            stddev = schedule(self.config.stddev_schedule, step)
            next_action = sample_noisy_action(
                next_action, stddev, self.config.stddev_clip
            )
            target_q1, target_q2 = self.critic_target(
                batch.next_states, next_action
            )
            target_q = batch.rewards + batch.discounts * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(batch.states, batch.actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        action = sample_noisy_action(
            self.actor(batch.states), stddev, self.config.stddev_clip
        )
        actor_q1, actor_q2 = self.critic(batch.states, action)
        actor_loss = actor_loss_from_q_values(actor_q1, actor_q2)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_parameter.lerp_(
                    parameter, self.config.critic_target_tau
                )

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q": float(q1.mean().item()),
            "target_q": float(target_q.mean().item()),
        }

    def save(self, path: str | Path, step: int) -> None:
        torch.save(
            {
                "observation_mode": "state",
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "step": int(step),
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> int:
        state = torch.load(path, map_location=self.device, weights_only=True)
        expected_metadata = {
            "observation_mode": "state",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        for field, expected in expected_metadata.items():
            actual = state.get(field)
            if actual != expected:
                raise ValueError(
                    f"Incompatible checkpoint {field}: expected {expected!r}, "
                    f"got {actual!r}"
                )
        checkpoint_config = state.get("config", {})
        checkpoint_hidden_dim = checkpoint_config.get("hidden_dim")
        if checkpoint_hidden_dim != self.config.hidden_dim:
            raise ValueError(
                "Incompatible checkpoint hidden_dim: expected "
                f"{self.config.hidden_dim!r}, got {checkpoint_hidden_dim!r}"
            )
        for name in ("actor", "critic", "critic_target"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        return int(state["step"])
