from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    pixels: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    next_pixels: torch.Tensor
    next_proprio: torch.Tensor


@dataclass(frozen=True)
class Transition:
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    discount: float
    next_observation: dict[str, np.ndarray]


class NStepAccumulator:
    def __init__(self, nstep: int, gamma: float):
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        self.nstep = int(nstep)
        self.gamma = float(gamma)
        self._pending: deque[tuple] = deque()

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: dict[str, np.ndarray],
        terminated: bool,
        episode_end: bool | None = None,
    ) -> list[Transition]:
        episode_end = terminated if episode_end is None else episode_end
        self._pending.append(
            (observation, np.asarray(action, dtype=np.float32).copy(), reward, next_observation, terminated)
        )
        transitions: list[Transition] = []
        if len(self._pending) >= self.nstep:
            transitions.append(self._build_transition(self.nstep))
            self._pending.popleft()
        if episode_end:
            while self._pending:
                transitions.append(self._build_transition(len(self._pending)))
                self._pending.popleft()
        return transitions

    def _build_transition(self, length: int) -> Transition:
        items = list(self._pending)[:length]
        reward = 0.0
        terminal = False
        for index, (_, _, step_reward, _, step_terminated) in enumerate(items):
            reward += self.gamma**index * float(step_reward)
            terminal = terminal or bool(step_terminated)
        observation, action, _, _, _ = items[0]
        next_observation = items[-1][3]
        discount = 0.0 if terminal else self.gamma**length
        return Transition(observation, action, reward, discount, next_observation)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.index = 0

    def _allocate(self, observation: dict[str, np.ndarray], action: np.ndarray) -> None:
        pixel_shape = observation["pixels"].shape
        proprio_shape = observation["proprio"].shape
        self.pixels = np.empty((self.capacity, *pixel_shape), dtype=np.uint8)
        self.proprio = np.empty((self.capacity, *proprio_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *action.shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.discounts = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_pixels = np.empty((self.capacity, 3, *pixel_shape[1:]), dtype=np.uint8)
        self.next_proprio = np.empty((self.capacity, *proprio_shape), dtype=np.float32)

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        discount: float,
        next_observation: dict[str, np.ndarray],
    ) -> None:
        action = np.asarray(action, dtype=np.float32)
        if not hasattr(self, "pixels"):
            self._allocate(observation, action)
        i = self.index
        self.pixels[i] = observation["pixels"]
        self.proprio[i] = observation["proprio"]
        self.actions[i] = action
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.next_pixels[i] = next_observation["pixels"][-3:]
        self.next_proprio[i] = next_observation["proprio"]
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError(f"Replay has {self.size} items, need {batch_size}")
        indices = self.rng.integers(0, self.size, size=batch_size)

        def tensor(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(values[indices], device=device)

        pixels = tensor(self.pixels)
        next_frame = tensor(self.next_pixels)
        return ReplayBatch(
            pixels=pixels,
            proprio=tensor(self.proprio),
            actions=tensor(self.actions),
            rewards=tensor(self.rewards),
            discounts=tensor(self.discounts),
            next_pixels=torch.cat((pixels[:, 3:], next_frame), dim=1),
            next_proprio=tensor(self.next_proprio),
        )

    def __len__(self) -> int:
        return self.size
